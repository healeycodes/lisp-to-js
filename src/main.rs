use pom::parser::*;
use std::{
    io::{stdin, Read},
    process::exit,
    str::{self, FromStr},
};

#[derive(Debug, PartialEq)]
enum Expression {
    Atom(Atom),
    List(Vec<Expression>),
    LetExpression(LetExpression),
    LambdaExpression(LambdaExpression),
    IfExpression(Box<IfExpression>),
    ArithmeticExpression(Box<ArithmeticExpression>),
}

#[derive(Debug, PartialEq)]
enum Atom {
    Number(f64),
    Symbol(String),
}

#[derive(Debug, PartialEq)]
struct LetExpression {
    bindings: Vec<Binding>,
    expressions: Vec<Expression>,
}

#[derive(Debug, PartialEq)]
struct Binding {
    symbol: String,
    expression: Expression,
}

#[derive(Debug, PartialEq)]
struct LambdaExpression {
    parameters: Vec<String>,
    expressions: Vec<Expression>,
}

#[derive(Debug, PartialEq)]
struct IfExpression {
    check: Expression,
    r#true: Expression,
    r#false: Expression,
}

#[derive(Debug, PartialEq)]
struct ArithmeticExpression {
    op: Op,
    expressions: Vec<Expression>,
}

#[derive(Debug, PartialEq)]
enum Op {
    Plus,
    Minus,
    LessThan,
    GreaterThan,
}

fn space<'a>() -> Parser<'a, u8, ()> {
    one_of(b" \t\r\n").repeat(0..).discard()
}

fn lparen<'a>() -> Parser<'a, u8, ()> {
    space() * seq(b"(").discard() - space()
}

fn rparen<'a>() -> Parser<'a, u8, ()> {
    space() * seq(b")").discard() - space()
}

fn number<'a>() -> Parser<'a, u8, f64> {
    let number = one_of(b"123456789") - one_of(b"0123456789").repeat(0..) | sym(b'0');
    number
        .collect()
        .convert(str::from_utf8)
        .convert(f64::from_str)
}

fn symbol<'a>() -> Parser<'a, u8, String> {
    space()
        * one_of(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            .repeat(1..)
            .convert(String::from_utf8)
        - space()
}

fn atom<'a>() -> Parser<'a, u8, Atom> {
    space() * (number().map(|n| Atom::Number(n)) | symbol().map(|s| Atom::Symbol(s))) - space()
}

fn expression<'a>() -> Parser<'a, u8, Expression> {
    space()
        * (((call(let_expression)
            | call(lambda_expression)
            | call(if_expression)
            | call(arithmetic_expression))
            | lparen() * call(expression).repeat(0..).map(Expression::List) - rparen())
            | atom().map(|a| Expression::Atom(a)))
        - space()
}

fn let_expression<'a>() -> Parser<'a, u8, Expression> {
    (lparen() - seq(b"let"))
        * (bindings() + expression().repeat(1..)).map(|(bindings, expressions)| {
            Expression::LetExpression(LetExpression {
                bindings,
                expressions,
            })
        })
        - rparen()
}

fn bindings<'a>() -> Parser<'a, u8, Vec<Binding>> {
    lparen()
        * ((lparen() * symbol() + expression() - rparen())
            .repeat(0..)
            .map(|bindings| {
                bindings
                    .into_iter()
                    .map(|(sym, expr)| Binding {
                        symbol: sym,
                        expression: expr,
                    })
                    .collect()
            }))
        - rparen()
}

fn lambda_expression<'a>() -> Parser<'a, u8, Expression> {
    ((lparen() + seq(b"lambda") + lparen()).discard() * symbol().repeat(0..) - rparen()
        + expression().repeat(1..)
        - rparen())
    .map(|(parameters, expressions)| {
        Expression::LambdaExpression(LambdaExpression {
            parameters,
            expressions,
        })
    })
}

fn if_expression<'a>() -> Parser<'a, u8, Expression> {
    ((lparen() + seq(b"if")).discard() * expression() + expression() + expression() - rparen()).map(
        |expressions| {
            Expression::IfExpression(Box::new(IfExpression {
                check: expressions.0 .0,
                r#true: expressions.0 .1,
                r#false: expressions.1,
            }))
        },
    )
}

fn arithmetic_expression<'a>() -> Parser<'a, u8, Expression> {
    (lparen() * one_of(b"+-<>") + expression().repeat(2..) - rparen()).map(|(op, expressions)| {
        Expression::ArithmeticExpression(Box::new(ArithmeticExpression {
            op: match op {
                b'+' => Op::Plus,
                b'-' => Op::Minus,
                b'<' => Op::LessThan,
                b'>' => Op::GreaterThan,
                _ => unreachable!(""),
            },
            expressions: expressions,
        }))
    })
}

fn program<'a>() -> Parser<'a, u8, Vec<Expression>> {
    expression().repeat(0..) - end()
}

fn compile_expression(expression: Expression) -> String {
    let mut ret = String::new();
    match expression {
        Expression::Atom(a) => match a {
            Atom::Number(n) => ret.push_str(&n.to_string()),
            Atom::Symbol(s) => ret.push_str(&format!(" {} ", &s.to_string())),
        },
        Expression::List(list) => {
            let mut i = 0;
            list.into_iter().for_each(|expression| {
                ret.push_str(&compile_expression(expression));
                if i == 0 {
                    ret.push_str("(")
                } else {
                    ret.push_str(", ")
                }
                i += 1;
            });
            if i > 0 {
                ret.push_str(")")
            }
        }
        Expression::LetExpression(let_expression) => {
            let mut bound_area = "(() => {\n".to_string();
            let_expression.bindings.into_iter().for_each(|binding| {
                bound_area.push_str(&format!(
                    "let {} = {};",
                    binding.symbol,
                    compile_expression(binding.expression)
                ));
            });
            let_expression
                .expressions
                .into_iter()
                .for_each(|expression| {
                    bound_area.push_str(&compile_expression(expression));
                });
            bound_area.push_str("\n})()");
            ret.push_str(&bound_area);
        }
        Expression::LambdaExpression(lambda_expression) => {
            let params = lambda_expression.parameters.join(",");
            let mut body = "".to_string();

            for expression in lambda_expression.expressions {
                body.push_str(&format!("{}\n", &compile_expression(expression)));
            }

            ret.push_str(&format!(" (({}) => {})\n", params, body));
        }
        Expression::IfExpression(if_expression) => ret.push_str(&format!(
            "{} ? {} : {}\n",
            compile_expression(if_expression.check),
            compile_expression(if_expression.r#true),
            compile_expression(if_expression.r#false)
        )),
        Expression::ArithmeticExpression(arithmetic_expression) => {
            let mut compiled_expressions: Vec<String> = vec![];
            for expression in arithmetic_expression.expressions {
                compiled_expressions.push(compile_expression(expression));
            }

            match arithmetic_expression.op {
                Op::Plus => {
                    ret.push_str("(");
                    ret.push_str(&compiled_expressions.join("+"));
                    ret.push_str(")");
                }
                Op::Minus => {
                    ret.push_str("(");
                    ret.push_str(&compiled_expressions.join("-"));
                    ret.push_str(")");
                }
                Op::LessThan => ret.push_str(
                    &compiled_expressions
                        .windows(2)
                        .into_iter()
                        .map(|expressions| expressions.join(" < "))
                        .collect::<Vec<String>>()
                        .join(" && "),
                ),
                Op::GreaterThan => ret.push_str(
                    &compiled_expressions
                        .windows(2)
                        .into_iter()
                        .map(|expressions| expressions.join(" > "))
                        .collect::<Vec<String>>()
                        .join(" && "),
                ),
            }
        }
    };
    ret
}

fn compile(program: Vec<Expression>) -> String {
    // Uncomment for debugging
    // println!("compiling: {:?}\n", program);

    let mut output = "/* lisp-to-js */
let print = console.log;


"
    .to_string();

    program.into_iter().for_each(|expression| {
        output.push_str(&compile_expression(expression));
    });

    output
}

fn main() {
    let mut buffer = Vec::new();
    stdin()
        .read_to_end(&mut buffer)
        .expect("error reading from stdin");

    let js_code = match (program()).parse(&buffer) {
        Err(e) => {
            eprintln!("{}", e);
            exit(1);
        }
        Ok(ast) => compile(ast),
    };

    println!("{}", js_code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_atom() {
        assert_eq!(
            format!("{:?}", program().parse(b"(1)").unwrap()),
            "[List([Atom(Number(1.0))])]"
        );
    }

    #[test]
    fn test_bindings_in_func() {
        assert_eq!(
            format!(
                "{:?}",
                compile(
                    program()
                        .parse(
                            b"(let ((a (lambda () (let ((b 1)) b)))) (print (a)))"
                        )
                        .unwrap()
                )
            ),
            "\"/* lisp-to-js */\\nlet print = console.log;\\n\\n\\n(() => {\\nlet a =  (() => (() => {\\nlet b = 1; b \\n})()\\n)\\n; print ( a (), )\\n})()\""
        );
    }

    #[test]
    fn test_compile_fib() {
        assert_eq!(
            format!(
                "{:?}",
                compile(
                    program()
                        .parse(
                            b"(let ((fib (lambda (n)
    (if (< n 2)
        n
        (+ (fib (- n 1)) (fib (- n 2)))))))
(print (fib 10)))"
                        )
                        .unwrap()
                )
            ),
            "\"/* lisp-to-js */\\nlet print = console.log;\\n\\n\\n(() => {\\nlet fib =  ((n) =>  n  < 2 ?  n  : ( fib (( n -1), )+ fib (( n -2), ))\\n\\n)\\n; print ( fib (10, ), )\\n})()\""
        );
    }
}

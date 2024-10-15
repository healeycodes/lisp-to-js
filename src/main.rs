use pom::parser::*;
use std::{
    cell::RefCell,
    collections::HashMap,
    env,
    error::Error,
    fmt,
    io::{stdin, Read},
    process::{self, exit},
    rc::Rc,
    str::{self, FromStr},
};

#[derive(Clone, Debug, PartialEq)]
enum Expression {
    Atom(Atom),
    List(Vec<Expression>),
    LetExpression(LetExpression),
    LambdaExpression(LambdaExpression),
    IfExpression(Box<IfExpression>),
    ArithmeticExpression(Box<ArithmeticExpression>),
}

#[derive(Clone, Debug, PartialEq)]
enum Atom {
    Boolean(bool),
    Number(f64),
    Symbol(String),
}

#[derive(Clone, Debug, PartialEq)]
struct LetExpression {
    bindings: Vec<Binding>,
    expressions: Vec<Expression>,
}

#[derive(Clone, Debug, PartialEq)]
struct Binding {
    symbol: String,
    expression: Expression,
}

#[derive(Clone, Debug, PartialEq)]
struct LambdaExpression {
    parameters: Vec<String>,
    expressions: Vec<Expression>,
}

#[derive(Clone, Debug, PartialEq)]
struct IfExpression {
    check: Expression,
    r#true: Expression,
    r#false: Expression,
}

#[derive(Clone, Debug, PartialEq)]
struct ArithmeticExpression {
    op: Op,
    expressions: Vec<Expression>,
}

#[derive(Clone, Debug, PartialEq)]
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
        * (bindings() + expression().repeat(0..)).map(|(bindings, expressions)| {
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
    let sum_or_difference = one_of(b"+-") + expression().repeat(2..);
    let lt_or_gt = one_of(b"<>") + expression().repeat(2);
    (lparen() * (sum_or_difference | lt_or_gt) - rparen()).map(|(op, expressions)| {
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
            Atom::Boolean(b) => match b {
                true => ret.push_str("true"),
                false => ret.push_str("false"),
            },
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
                Op::LessThan => {
                    ret.push_str(&compiled_expressions.join(" < "));
                }
                Op::GreaterThan => {
                    ret.push_str(&compiled_expressions.join(" > "));
                }
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

fn optimize(program: Vec<Expression>) -> Vec<Expression> {
    return program
        .into_iter()
        .map(|expr| optimize_expression(expr, &mut HashMap::new()))
        .collect();
}

// get_expr_from_context returns an atom mber from a context if it exists
fn get_expr_from_context(
    symbol: String,
    context: &HashMap<String, Option<Expression>>,
) -> Option<Atom> {
    match context.get(&symbol) {
        Some(expr) => match expr {
            Some(expr) => match expr {
                Expression::Atom(atom) => match atom {
                    Atom::Number(n) => Some(Atom::Number(*n)),
                    Atom::Boolean(b) => Some(Atom::Boolean(*b)),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        },
        None => None,
    }
}

fn optimize_expression(
    expression: Expression,
    context: &mut HashMap<String, Option<Expression>>,
) -> Expression {
    match expression {
        // Only internal optimizations are possible
        Expression::List(list_expr) => {
            return Expression::List(
                list_expr
                    .into_iter()
                    .map(|expr| optimize_expression(expr, context))
                    .collect(),
            )
        }

        // Only the internals of let expressions can be optimized.
        // The bindings can be reduced to an empty list of bindings if they all fold into number assignments
        // (let (a 1) a) -> (let () 1)
        Expression::LetExpression(let_expr) => {
            let mut optimized_bindings: Vec<Binding> = vec![];

            let_expr.bindings.into_iter().for_each(|binding| {
                let binding_expr = optimize_expression(binding.expression, context);

                // When the expression we're about to bind is an atom, we can optimize the binding away
                match binding_expr {
                    Expression::Atom(ref atom) => match atom {
                        // Insert literals, overwriting variables from any higher scopes.
                        // Return before pushing the binding so it's removed from the AST
                        Atom::Number(n) => {
                            context
                                .insert(binding.symbol, Some(Expression::Atom(Atom::Number(*n))));
                            return;
                        }
                        Atom::Boolean(b) => {
                            context
                                .insert(binding.symbol, Some(Expression::Atom(Atom::Boolean(*b))));
                            return;
                        }

                        // No need to overwrite symbols that refer to already-tracked and potentially optimized values
                        Atom::Symbol(s) => match context.get(s) {
                            Some(_) => return,
                            None => {}
                        },
                    },
                    _ => {}
                }

                // This binding can't be removed but may have been optimized internally
                optimized_bindings.push(Binding {
                    symbol: binding.symbol,
                    expression: binding_expr,
                })
            });

            return Expression::LetExpression(LetExpression {
                bindings: optimized_bindings,
                expressions: let_expr
                    .expressions
                    .into_iter()
                    .map(|expr| optimize_expression(expr, context))
                    .collect(),
            });
        }

        // Only internal optimizations are possible
        Expression::LambdaExpression(lambda_expr) => {
            Expression::LambdaExpression(LambdaExpression {
                parameters: lambda_expr.parameters,
                expressions: lambda_expr
                    .expressions
                    .into_iter()
                    .map(|expr| optimize_expression(expr, context))
                    .collect(),
            })
        }

        // The goal with if expressions is to remove the check and replace it with the winning branch
        Expression::IfExpression(if_expr) => {
            let check_expr = optimize_expression(if_expr.check, context);
            match check_expr {
                Expression::Atom(ref atom) => match atom {
                    Atom::Boolean(b) => {
                        if *b {
                            return optimize_expression(if_expr.r#true, context);
                        } else {
                            return optimize_expression(if_expr.r#false, context);
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
            return Expression::IfExpression(Box::new(IfExpression {
                check: optimize_expression(check_expr, context),
                r#true: optimize_expression(if_expr.r#true, context),
                r#false: optimize_expression(if_expr.r#false, context),
            }));
        }

        // Arithmetic expressions can be replaced with atoms or reduced
        // (+ 1 2) -> 3
        // (+ 1 a 2) -> (+ a 3)
        // (< 1 2) -> true
        // (< 1 2 a) -> (< 1 a)
        Expression::ArithmeticExpression(arth_expr) => {
            let optimized_exprs: Vec<Expression> = arth_expr
                .expressions
                .into_iter()
                .map(|expr| optimize_expression(expr, context))
                .collect();

            if optimized_exprs.len() < 2 {
                unreachable!("parser (should) assume arithmetic expressions contain 2+ items")
            }

            let mut nums: Vec<f64> = vec![];
            let mut optimized_exprs_without_numbers: Vec<Expression> = vec![];
            for expr in &optimized_exprs {
                match expr {
                    Expression::Atom(atom) => match atom {
                        Atom::Number(n) => nums.push(*n),
                        Atom::Boolean(b) => optimized_exprs_without_numbers
                            .push(Expression::Atom(Atom::Boolean(*b))),
                        Atom::Symbol(s) => match get_expr_from_context(s.to_string(), context) {
                            Some(atom) => match atom {
                                Atom::Number(n) => nums.push(n),
                                Atom::Boolean(_) => unreachable!("parser (should) have stopped a bool from entering an arithmetic expression"),
                                Atom::Symbol(_) => unreachable!("optimizer shouldn't insert symbols into context"),
                            },
                            _ => optimized_exprs_without_numbers
                                .push(Expression::Atom(Atom::Symbol(s.to_string()))),
                        },
                    },
                    Expression::List(list_expr) => {
                        optimized_exprs_without_numbers.push(Expression::List(list_expr.to_vec()))
                    }
                    Expression::LetExpression(let_expr) => optimized_exprs_without_numbers
                        .push(Expression::LetExpression(let_expr.clone())),
                    Expression::LambdaExpression(lambda_expr) => optimized_exprs_without_numbers
                        .push(Expression::LambdaExpression(lambda_expr.clone())),
                    Expression::IfExpression(if_expr) => optimized_exprs_without_numbers
                        .push(Expression::IfExpression(if_expr.clone())),
                    Expression::ArithmeticExpression(arth_expr) => optimized_exprs_without_numbers
                        .push(Expression::ArithmeticExpression(arth_expr.clone())),
                }
            }

            // When there are no literals (after optimization) we just return as-is
            if nums.len() == 0 {
                return Expression::ArithmeticExpression(Box::new(ArithmeticExpression {
                    op: arth_expr.op,
                    expressions: optimized_exprs,
                }));
            }

            match arth_expr.op {
                Op::Plus => {
                    // Best case: no expressions after optimization, return atom
                    if optimized_exprs_without_numbers.len() == 0 {
                        return Expression::Atom(Atom::Number(nums.iter().sum()));
                    }

                    // Sum any literals, may reduce add-operations produced at code generation
                    optimized_exprs_without_numbers
                        .push(Expression::Atom(Atom::Number(nums.iter().sum())));
                    return Expression::ArithmeticExpression(Box::new(ArithmeticExpression {
                        op: arth_expr.op,
                        expressions: optimized_exprs_without_numbers,
                    }));
                }
                Op::Minus => {
                    // Note: the minus expression isn't "negate all numbers"
                    // it's minus all numbers from the first
                    let first = *nums.first().unwrap_or(&0.0);
                    let compressed =
                        Atom::Number(nums.iter().skip(1).fold(first, |acc, &x| acc - x));

                    // Best case: no expressions after optimization, return atom
                    if optimized_exprs_without_numbers.len() == 0 {
                        return Expression::Atom(compressed);
                    }

                    optimized_exprs_without_numbers.push(Expression::Atom(compressed));
                    return Expression::ArithmeticExpression(Box::new(ArithmeticExpression {
                        op: arth_expr.op,
                        expressions: optimized_exprs_without_numbers,
                    }));
                }
                Op::LessThan | Op::GreaterThan => {
                    let compare_func: fn(f64, f64) -> bool = match arth_expr.op {
                        Op::LessThan => lt,
                        Op::GreaterThan => gt,
                        _ => unreachable!(),
                    };

                    // Best case: after optimization the expression is redundant
                    if optimized_exprs_without_numbers.len() == 0 {
                        if nums.len() != 2 {
                            unreachable!("parser (should) have ensured two expressions");
                        }

                        return Expression::Atom(Atom::Boolean(compare_func(nums[0], nums[1])));
                    };

                    return Expression::ArithmeticExpression(Box::new(ArithmeticExpression {
                        op: arth_expr.op,
                        expressions: optimized_exprs,
                    }));
                }
            }
        }
        Expression::Atom(ref atom) => match atom {
            Atom::Symbol(s) => match get_expr_from_context(s.to_string(), context) {
                Some(atom) => return Expression::Atom(atom),
                _ => return expression,
            },
            _ => return expression,
        },
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ByteCodeInstruction {
    PushConst(f64),    // Pushes a constant (float) onto the stack
    PushBool(bool),    // Pushes a boolean onto the stack
    LoadVar(String),   // Loads a variable by its name
    StoreVar(String),  // Stores the top of the stack into a variable
    Add(usize),        // Adds n numbers on the stack
    Sub(usize),        // Subtracts n numbers on the stack
    LessThan,          // Compares top two stack values (x < y)
    GreaterThan,       // Compares top two stack values (x > y)
    Jump(usize),       // Unconditional jump
    CallLambda(usize), // Calls a lambda expression with n arguments
    PushClosure(Vec<String>, Vec<ByteCodeInstruction>), // Pushes a closure onto the stack
}

fn compile_byte_code(expressions: Vec<Expression>) -> Vec<ByteCodeInstruction> {
    let mut bytecode = Vec::new();
    for expr in expressions {
        compile_byte_code_expression(&expr, &mut bytecode);
    }
    bytecode
}

fn compile_byte_code_expression(expr: &Expression, bytecode: &mut Vec<ByteCodeInstruction>) {
    match expr {
        Expression::Atom(atom) => compile_byte_code_atom(atom, bytecode),
        Expression::List(list) => compile_byte_code_list(list, bytecode),
        Expression::LetExpression(let_expr) => compile_byte_code_let(let_expr, bytecode),
        Expression::LambdaExpression(lambda_expr) => {
            compile_byte_code_lambda(lambda_expr, bytecode)
        }
        Expression::IfExpression(if_expr) => compile_byte_code_if(if_expr, bytecode),
        Expression::ArithmeticExpression(arith_expr) => {
            compile_byte_code_arithmetic(arith_expr, bytecode)
        }
    }
}

fn compile_byte_code_atom(atom: &Atom, bytecode: &mut Vec<ByteCodeInstruction>) {
    match atom {
        Atom::Boolean(val) => bytecode.push(ByteCodeInstruction::PushBool(*val)),
        Atom::Number(num) => bytecode.push(ByteCodeInstruction::PushConst(*num)),
        Atom::Symbol(sym) => bytecode.push(ByteCodeInstruction::LoadVar(sym.clone())),
    }
}

fn compile_byte_code_list(list: &Vec<Expression>, bytecode: &mut Vec<ByteCodeInstruction>) {
    if let Some((first, rest)) = list.split_first() {
        // Compile arguments first
        for expr in rest {
            compile_byte_code_expression(expr, bytecode);
        }

        // Compile the function expression (e.g., a symbol for `print`)
        compile_byte_code_expression(first, bytecode);

        // Emit CallLambda with the number of arguments
        bytecode.push(ByteCodeInstruction::CallLambda(rest.len()));
    }
}

fn compile_byte_code_let(let_expr: &LetExpression, bytecode: &mut Vec<ByteCodeInstruction>) {
    // Compile the bindings: store expressions in the variables
    for binding in &let_expr.bindings {
        compile_byte_code_expression(&binding.expression, bytecode);
        bytecode.push(ByteCodeInstruction::StoreVar(binding.symbol.clone()));
    }

    // Compile the expressions within the `let` body
    for expr in &let_expr.expressions {
        compile_byte_code_expression(expr, bytecode);
    }
}

fn compile_byte_code_lambda(
    lambda_expr: &LambdaExpression,
    bytecode: &mut Vec<ByteCodeInstruction>,
) {
    // Capture free variables (we assume all parameters are bound)
    let mut closure_bytecode = Vec::new();

    for expr in &lambda_expr.expressions {
        compile_byte_code_expression(expr, &mut closure_bytecode);
    }

    // Push a closure with its captured parameters and bytecode
    bytecode.push(ByteCodeInstruction::PushClosure(
        lambda_expr.parameters.clone(),
        closure_bytecode,
    ));
}

fn compile_byte_code_if(if_expr: &IfExpression, bytecode: &mut Vec<ByteCodeInstruction>) {
    // Compile the condition expression
    compile_byte_code_expression(&if_expr.check, bytecode);

    // Placeholder index for jump_if_false, to be patched later
    let jump_if_false_pos = bytecode.len();
    bytecode.push(ByteCodeInstruction::Jump(0)); // Placeholder for jump to start of false branch if condition is false

    // Compile the true branch
    compile_byte_code_expression(&if_expr.r#true, bytecode);

    // Placeholder index for jump_over_false, to skip the false branch after true branch is executed
    let jump_over_false_pos = bytecode.len();
    bytecode.push(ByteCodeInstruction::Jump(0)); // Placeholder for jump over false branch after true branch

    // Patch the jump_if_false instruction to jump to the false branch
    let false_branch_pos = bytecode.len();
    if let ByteCodeInstruction::Jump(ref mut target) = bytecode[jump_if_false_pos] {
        *target = false_branch_pos;
    }

    // Compile the false branch
    compile_byte_code_expression(&if_expr.r#false, bytecode);

    // Patch the jump instruction to jump past the false branch
    let end_pos = bytecode.len();
    if let ByteCodeInstruction::Jump(ref mut target) = bytecode[jump_over_false_pos] {
        *target = end_pos;
    }
}

fn compile_byte_code_arithmetic(
    arith_expr: &ArithmeticExpression,
    bytecode: &mut Vec<ByteCodeInstruction>,
) {
    // Compile each expression in the arithmetic expression
    for expr in &arith_expr.expressions {
        compile_byte_code_expression(expr, bytecode);
    }

    // Emit the appropriate arithmetic operation
    match arith_expr.op {
        Op::Plus => bytecode.push(ByteCodeInstruction::Add(arith_expr.expressions.len())),
        Op::Minus => bytecode.push(ByteCodeInstruction::Sub(arith_expr.expressions.len())),
        Op::LessThan => bytecode.push(ByteCodeInstruction::LessThan),
        Op::GreaterThan => bytecode.push(ByteCodeInstruction::GreaterThan),
    }
}

fn debug_byte_code(byte_code: Vec<ByteCodeInstruction>, depth: usize) -> String {
    let mut output = String::new();

    for (index, instruction) in byte_code.iter().enumerate() {
        if depth > 0 {
            for _ in 0..depth {
                output.push_str("    ");
            }
            output.push_str("->")
        }
        match instruction {
            ByteCodeInstruction::PushConst(value) => {
                output.push_str(&format!("{:>4}: push_const {:.1}\n", index, value));
            }
            ByteCodeInstruction::PushBool(value) => {
                output.push_str(&format!("{:>4}: push_bool {}\n", index, value));
            }
            ByteCodeInstruction::LoadVar(var_name) => {
                output.push_str(&format!("{:>4}: load_var {}\n", index, var_name));
            }
            ByteCodeInstruction::StoreVar(var_name) => {
                output.push_str(&format!("{:>4}: store_var {}\n", index, var_name));
            }
            ByteCodeInstruction::Add(n) => {
                output.push_str(&format!("{:>4}: add {}\n", index, n));
            }
            ByteCodeInstruction::Sub(n) => {
                output.push_str(&format!("{:>4}: sub {}\n", index, n));
            }
            ByteCodeInstruction::LessThan => {
                output.push_str(&format!("{:>4}: less_than\n", index));
            }
            ByteCodeInstruction::GreaterThan => {
                output.push_str(&format!("{:>4}: greater_than\n", index));
            }
            ByteCodeInstruction::Jump(target) => {
                let detail = match target >= &byte_code.len() {
                    true => "exit",
                    false => &format!("go to {}", target),
                };
                output.push_str(&format!("{:>4}: jump {} // {}\n", index, target, detail));
            }
            ByteCodeInstruction::CallLambda(arg_count) => {
                output.push_str(&format!("{:>4}: call_lambda {}\n", index, arg_count));
            }
            ByteCodeInstruction::PushClosure(params, instructions) => {
                output.push_str(&format!("{:>4}: push_closure {:?}\n", index, params));
                output.push_str(&debug_byte_code(instructions.to_vec(), depth + 1));
            }
        }
    }

    output
}

#[derive(Debug)]
struct RuntimeError {
    details: String,
}

impl RuntimeError {
    fn new(msg: &str) -> RuntimeError {
        RuntimeError {
            details: msg.to_string(),
        }
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for RuntimeError {
    fn description(&self) -> &str {
        &self.details
    }
}

#[derive(Debug, Clone)]
enum StackValue {
    Number(f64),
    Bool(bool),
    Closure(Vec<String>, Vec<ByteCodeInstruction>),
    BuiltinFunction(fn(Vec<StackValue>) -> Result<(), RuntimeError>),
}

#[derive(Clone)]
struct StackFrame {
    parent: Option<Rc<RefCell<StackFrame>>>,
    variables: HashMap<String, StackValue>,
}

impl StackFrame {
    fn new() -> Rc<RefCell<StackFrame>> {
        Rc::new(RefCell::new(StackFrame {
            parent: None,
            variables: HashMap::new(),
        }))
    }

    fn child(parent: Rc<RefCell<StackFrame>>) -> Rc<RefCell<StackFrame>> {
        Rc::new(RefCell::new(StackFrame {
            parent: Some(parent),
            variables: HashMap::new(),
        }))
    }

    fn store(&mut self, variable: String, value: StackValue) {
        self.variables.insert(variable, value);
    }

    fn look_up(&self, variable: &str) -> Result<StackValue, RuntimeError> {
        match self.variables.get(variable) {
            Some(value) => Ok(value.clone()),
            None => match &self.parent {
                Some(parent) => parent.borrow().look_up(variable),
                None => Err(RuntimeError::new(&format!(
                    "unknown variable: {}",
                    variable
                ))),
            },
        }
    }
}

fn print_builtin(args: Vec<StackValue>) -> Result<(), RuntimeError> {
    for arg in args {
        match arg {
            StackValue::Number(n) => print!("{}", n),
            StackValue::Bool(b) => print!("{}", b),
            StackValue::Closure(_, _) => print!("closure"),
            StackValue::BuiltinFunction(_) => print!("built-in"),
        }
    }
    println!();
    Ok(())
}

fn byte_code_vm(
    byte_code: Vec<ByteCodeInstruction>,
    stack: &mut Vec<StackValue>,
    frame: Rc<RefCell<StackFrame>>,
) -> Result<(), RuntimeError> {
    let mut position: usize = 0;

    while position < byte_code.len() {
        let instruction = &byte_code[position];

        // // Uncomment for debugging (TODO: use verbose debug flag?)
        // println!(
        //     "{}",
        //     format!(
        //         "{}: {:?} stack: {:?} frame: {:?}",
        //         position,
        //         instruction,
        //         stack,
        //         frame.borrow().variables
        //     )
        // );

        match instruction {
            ByteCodeInstruction::PushConst(value) => stack.push(StackValue::Number(*value)),
            ByteCodeInstruction::PushBool(value) => stack.push(StackValue::Bool(*value)),
            ByteCodeInstruction::LoadVar(variable) => {
                if variable == "print" {
                    stack.push(StackValue::BuiltinFunction(print_builtin));
                } else {
                    stack.push(frame.borrow().look_up(variable)?)
                }
            }
            ByteCodeInstruction::StoreVar(variable) => match stack.pop() {
                None => {
                    return Err(RuntimeError::new(&format!(
                        "stack empty when setting {}",
                        variable
                    )))
                }
                Some(value) => frame.borrow_mut().store(variable.to_string(), value),
            },
            ByteCodeInstruction::Add(n) => {
                if stack.len() < *n {
                    return Err(RuntimeError::new(&format!(
                        "not enough items on the stack (unreachable)"
                    )));
                }
                let values: Vec<StackValue> = stack.split_off(stack.len() - n);

                if values.iter().any(|v| matches!(v, StackValue::Bool(_))) {
                    return Err(RuntimeError::new(&format!("cannot add boolean")));
                }

                let sum: f64 = values
                    .into_iter()
                    .map(|v| match v {
                        StackValue::Number(num) => num,
                        _ => unreachable!(),
                    })
                    .sum();

                stack.push(StackValue::Number(sum));
            }
            ByteCodeInstruction::Sub(n) => {
                if stack.len() < *n {
                    return Err(RuntimeError::new(&format!(
                        "not enough items on the stack (unreachable)"
                    )));
                }
                let values: Vec<StackValue> = stack.split_off(stack.len() - n);

                if values.iter().any(|v| matches!(v, StackValue::Bool(_))) {
                    return Err(RuntimeError::new(&format!("cannot subtract boolean")));
                }

                let mut iter = values.into_iter().map(|v| match v {
                    StackValue::Number(num) => num,
                    _ => unreachable!(),
                });

                let first = iter.next().unwrap();
                let difference: f64 = iter.fold(first, |acc, num| acc - num);

                stack.push(StackValue::Number(difference));
            }
            ByteCodeInstruction::LessThan => {
                if stack.len() < 2 {
                    return Err(RuntimeError::new("not enough items on the stack"));
                }

                let values = stack.split_off(stack.len() - 2);

                let (num1, num2) = match (&values[0], &values[1]) {
                    (StackValue::Number(n1), StackValue::Number(n2)) => (n1, n2),
                    _ => return Err(RuntimeError::new("cannot compare non-numeric values")),
                };

                if num1 < num2 {
                    position += 1;
                }
            }
            ByteCodeInstruction::GreaterThan => {
                if stack.len() < 2 {
                    return Err(RuntimeError::new("not enough items on the stack"));
                }

                let values = stack.split_off(stack.len() - 2);

                let (num1, num2) = match (&values[0], &values[1]) {
                    (StackValue::Number(n1), StackValue::Number(n2)) => (n1, n2),
                    _ => return Err(RuntimeError::new("cannot compare non-numeric values")),
                };

                if num1 > num2 {
                    position += 1;
                }
            }
            ByteCodeInstruction::Jump(new_position) => {
                position = *new_position;
                continue;
            }
            ByteCodeInstruction::CallLambda(n) => {
                if stack.len() <= *n {
                    return Err(RuntimeError::new("not enough items on the stack"));
                }

                match stack.pop() {
                    Some(value) => match value {
                        StackValue::Closure(params, closure_byte_code) => {
                            let child_frame = StackFrame::child(Rc::clone(&frame));

                            // Retrieve the arguments from the stack
                            let args = stack.split_off(stack.len() - n);
                            for (param, arg) in params.iter().zip(args) {
                                child_frame.borrow_mut().store(param.clone(), arg);
                            }

                            byte_code_vm(closure_byte_code, stack, Rc::clone(&child_frame))?;
                        }
                        StackValue::BuiltinFunction(func) => {
                            let args = stack.split_off(stack.len() - n);
                            func(args)?;
                        }
                        _ => {
                            return Err(RuntimeError::new(
                                "cannot call non-closure or non-function",
                            ))
                        }
                    },
                    None => unreachable!(),
                }
            }
            ByteCodeInstruction::PushClosure(params, closure_byte_code) => stack.push(
                StackValue::Closure(params.to_vec(), closure_byte_code.clone()),
            ),
        }

        position += 1;
    }

    Ok(())
}

fn main() {
    let mut buffer = Vec::new();
    stdin()
        .read_to_end(&mut buffer)
        .expect("error reading from stdin");

    let mut expressions = match (program()).parse(&buffer) {
        Err(e) => {
            eprintln!("{}", e);
            exit(1);
        }
        Ok(ast) => ast,
    };

    let args: Vec<String> = env::args().collect();
    if args.contains(&"--optimize".to_string()) {
        expressions = optimize(expressions);
    }

    if args.contains(&"--vm".to_string()) {
        let byte_code = compile_byte_code(expressions);

        if args.contains(&"--debug".to_string()) {
            println!("{}", debug_byte_code(byte_code.clone(), 0));
        }

        match byte_code_vm(byte_code, &mut vec![], StackFrame::new()) {
            Err(err) => {
                eprintln!("{}", format!("{}", err));
                std::process::exit(1);
            }
            _ => {}
        }
    } else if args.contains(&"--js".to_string()) {
        println!("{}", compile(expressions));
    } else {
        println!("must pass '--vm' or '--js'");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_code_compile() {
        let compiled = compile_byte_code(
            program()
                .parse(
                    b"(let ((fib (lambda (n)
    (if (< n 2)
        n
        (+ (fib (- n 1)) (fib (- n 2)))))))
(print (fib 10)))",
                )
                .unwrap(),
        );
        assert_eq!(format!("{:?}", compiled), "[PushClosure([\"n\"], [LoadVar(\"n\"), PushConst(2.0), LessThan, Jump(6), LoadVar(\"n\"), Jump(17), LoadVar(\"n\"), PushConst(1.0), Sub(2), LoadVar(\"fib\"), CallLambda(1), LoadVar(\"n\"), PushConst(2.0), Sub(2), LoadVar(\"fib\"), CallLambda(1), Add(2)]), StoreVar(\"fib\"), PushConst(10.0), LoadVar(\"fib\"), CallLambda(1), LoadVar(\"print\"), CallLambda(1)]");
    }

    #[test]
    fn test_byte_code_vm_arithmetic() {
        let compiled = compile_byte_code(program().parse(b"(+ 1 2 (- 5 100))").unwrap());

        let mut stack = vec![];
        byte_code_vm(compiled, &mut stack, StackFrame::new()).unwrap();
        assert_eq!(format!("{:?}", stack), "[Number(-92.0)]");
    }

    #[test]
    fn test_byte_code_vm_fib() {
        let compiled = compile_byte_code(
            program()
                .parse(
                    b"(let ((fib (lambda (n)
    (if (< n 2)
        n
        (+ (fib (- n 1)) (fib (- n 2)))))))
(fib 10))",
                )
                .unwrap(),
        );

        let mut stack = vec![];
        byte_code_vm(compiled, &mut stack, StackFrame::new()).unwrap();
        assert_eq!(format!("{:?}", stack), "[Number(55.0)]");
    }

    #[test]
    fn test_byte_code_vm_fib_print() {
        let compiled = compile_byte_code(
            program()
                .parse(
                    b"(let ((fib (lambda (n)
    (if (< n 2)
        n
        (+ (fib (- n 1)) (fib (- n 2)))))))
(print (fib 10)))",
                )
                .unwrap(),
        );

        let mut stack = vec![];
        byte_code_vm(compiled, &mut stack, StackFrame::new()).unwrap();

        // Empty because print pops it
        assert_eq!(format!("{:?}", stack), "[]");
    }

    #[test]
    fn test_byte_code_vm_double() {
        let compiled = compile_byte_code(
            program()
                .parse(b"(let ((double (lambda (x) (+ x x)))) (double 2))")
                .unwrap(),
        );

        let mut stack = vec![];
        byte_code_vm(compiled, &mut stack, StackFrame::new()).unwrap();
        assert_eq!(format!("{:?}", stack), "[Number(4.0)]");
    }

    #[test]
    fn test_byte_code_vm_nested_lambda() {
        let compiled = compile_byte_code(
            program()
                .parse(
                    b"(let ((fib (lambda (n)
    (if (< n 2)
        (let ((double (lambda (x) (+ x x)))) (double n))
        (+ (fib (- n 1)) (fib (- n 2)))))))
(fib 10))",
                )
                .unwrap(),
        );

        let mut stack = vec![];
        byte_code_vm(compiled, &mut stack, StackFrame::new()).unwrap();
        assert_eq!(format!("{:?}", stack), "[Number(110.0)]");
    }

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

    #[test]
    fn test_optimize_add() {
        assert_eq!(
            format!("{:?}", optimize(program().parse(b"(+ 1 2)").unwrap())),
            "[Atom(Number(3.0))]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(program().parse(b"(let ((a 1)) (+ 1 a 2))").unwrap())
            ),
            "[LetExpression(LetExpression { bindings: [], expressions: [Atom(Number(4.0))] })]"
        );
    }

    #[test]
    fn test_optimize_sub() {
        assert_eq!(
            format!("{:?}", optimize(program().parse(b"(- 1 2)").unwrap())),
            "[Atom(Number(-1.0))]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(program().parse(b"(let ((a 1)) (- 1 a 2))").unwrap())
            ),
            "[LetExpression(LetExpression { bindings: [], expressions: [Atom(Number(-2.0))] })]"
        );
    }

    #[test]
    fn test_optimize_many() {
        assert_eq!(
            format!(
                "{:?}",
                optimize(program().parse(b"(if (< 2 1) 1 2)").unwrap())
            ),
            "[Atom(Number(2.0))]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(program().parse(b"(if (> 2 1) 1 2)").unwrap())
            ),
            "[Atom(Number(1.0))]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(
                    program()
                        .parse(
                            b"(let ((a 1)) (let ((a 2)) a))"
                        )
                        .unwrap()
                )
            ),
            "[LetExpression(LetExpression { bindings: [], expressions: [LetExpression(LetExpression { bindings: [], expressions: [Atom(Number(2.0))] })] })]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(
                    program()
                        .parse(
                            b"(let ((a 5) (b (+ a 5)) (c 10)) 
                (print (if (> c 5) (+ a b) (- c 1))))"
                        )
                        .unwrap()
                )
            ),
            "[LetExpression(LetExpression { bindings: [], expressions: [List([Atom(Symbol(\"print\")), Atom(Number(15.0))])] })]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(
                    program()
                        .parse(
                            b"(lambda (a) (let (a 2) (+ a 2 2)) (+ 5 5))"
                        )
                        .unwrap()
                )
            ),
            "[LambdaExpression(LambdaExpression { parameters: [\"a\"], expressions: [List([Atom(Symbol(\"let\")), List([Atom(Symbol(\"a\")), Atom(Number(2.0))]), ArithmeticExpression(ArithmeticExpression { op: Plus, expressions: [Atom(Symbol(\"a\")), Atom(Number(4.0))] })]), Atom(Number(10.0))] })]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(program().parse(b"(let ((a (if (< 1 2) 1 2))) a)").unwrap())
            ),
            "[LetExpression(LetExpression { bindings: [], expressions: [Atom(Number(1.0))] })]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(program().parse(b"(let ((a 1)) (< 1 a))").unwrap())
            ),
            "[LetExpression(LetExpression { bindings: [], expressions: [Atom(Boolean(false))] })]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(program().parse(b"(let ((a (+ 1 2))) (< 1 a))").unwrap())
            ),
            "[LetExpression(LetExpression { bindings: [], expressions: [Atom(Boolean(true))] })]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(program().parse(b"(< a 1)").unwrap())
            ),
            "[ArithmeticExpression(ArithmeticExpression { op: LessThan, expressions: [Atom(Symbol(\"a\")), Atom(Number(1.0))] })]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(program().parse(b"(let ((a (+ 1 2)))
                (print (+ a a))
              )").unwrap())
            ),
            "[LetExpression(LetExpression { bindings: [], expressions: [List([Atom(Symbol(\"print\")), Atom(Number(6.0))])] })]"
        );
        assert_eq!(
            format!(
                "{:?}",
                optimize(program().parse(b"(let ((a (< 1 2))) a)").unwrap())
            ),
            "[LetExpression(LetExpression { bindings: [], expressions: [Atom(Boolean(true))] })]"
        );
    }
}

fn gt<T: PartialOrd>(a: T, b: T) -> bool {
    a > b
}

fn lt<T: PartialOrd>(a: T, b: T) -> bool {
    a < b
}

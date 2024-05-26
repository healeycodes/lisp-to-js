# lisp-to-js

This compiler translates Lisp into JavaScript.

It supports a Lisp that is very similar to Little Lisp.

```lisp
; atoms
1 ; f64 numbers
a ; symbols

; arithmetic expressions
(+ 1 2) ; 3
(- 1 2) ; -1

; control flow expressions
(< 1 2) ; true
(> 1 2) ; false
(if (< 1 2) (+ 10 10) (+ 10 5)) ; 20

; lambda expressions
(lambda (x) (+ x x)) ; function that doubles

; variable definition
(let ((a 1)) (print a)) ; prints 1
(let ((double (lambda (x) (+ x x))) (double 2))) ; 4
```

It produces ES6 JavaScript. Here's an example:

```js
/*
(let ((fib (lambda (n)
    (if (< n 2)
        n
        (+ (fib (- n 1)) (fib (- n 2)))))))
(print (fib 10)))
*/

let print = console.log;

let fib =  ((n) =>  n  < 2 ?  n  : ( fib (( n -1), )+ fib (( n -2), ))

)
; print ( fib (10, ), )
```

If you run this output through a tool like Prettier, you can see it's actually fairly sensible JavaScript:

```js
let print = console.log;

let fib = (n) => (n < 2 ? n : fib(n - 1) + fib(n - 2));
print(fib(10));
```

<br>

The parser is written using the [pom](https://github.com/J-F-Liu/pom) library.
- Transforms the AST into JavaScript code, maintaining the structure and logic of the original Lisp code.
- Outputs JavaScript code that can be executed in any JavaScript environment.

<br>

### Running the Compiler

To compile a Lisp file to JavaScript, run the following command:

```bash
$ cargo run < fib10.lisp > fib10.js
$ node fib10.js
55
```

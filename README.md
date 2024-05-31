# lisp-to-js

> My blog posts:
>
> - [Lisp to JavaScript Compiler](https://healeycodes.com/lisp-to-javascript-compiler)
> - [Lisp Compiler Optimizations](https://healeycodes.com/lisp-compiler-optimizations)

<br>

This compiler optimizes Lisp code and turns it into JavaScript.

Here's an example:

```js
/*
(let ((fib (lambda (n)
    (if (< n 2)
        n
        (+ (fib (- n 1)) (fib (- n 2)))))))
(print (fib 10)))
*/

let print = console.log;

let fib = (n) => n < 2 ? n : (fib(n - 1) + fib(n - 2));
print(fib(10));
```

<br>

The implemented optimizations are constant folding and propagation, and dead code elimination. This stage runs before code generation.

```lisp
; before optimization
(let ((b 2) (c 3))
  (print
    (+
      (+ b 4 c)
      (- b c 7)
  )))
 
; after optimization
(let () (print 1))
```

<br>

The Lisp variant is very similar to
[Little Lisp](https://maryrosecook.com/blog/post/little-lisp-interpreter).

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
(let ((double (lambda (x) (+ x x)))) (double 2)) ; 4
```

<br>

### Tests

There are tests for parsing, optimization, and code generation. 

Run:

```bash
cargo test
```

<br>

### Running the Compiler

To compile Lisp code to JavaScript, run the following command:

```bash
$ cargo run -- -o < fib10.lisp > fib10.js
$ node fib10.js
55
```

Flags:

- `-o` for optimization

# lisp-to-js

> My blog posts:
>
> - [Compiling Lisp to Bytecode and Running It](https://healeycodes.com/compiling-lisp-to-bytecode-and-running-it)
> - [Lisp Compiler Optimizations](https://healeycodes.com/lisp-compiler-optimizations)
> - [Lisp to JavaScript Compiler](https://healeycodes.com/lisp-to-javascript-compiler)

<br>

This project is an optmizing Lisp compiler and bytecode VM.

It can compile to JavaScript, or compile to bytecode and execute in a VM.

<br>

Bytecode VM:

```
./program --vm --debug < fib10.lisp
   0: push_closure ["n"]
    ->   0: load_var n
    ->   1: push_const 2.0
    ->   2: less_than
    ->   3: jump 6 // go to 6
    ->   4: load_var n
    ->   5: jump 17 // exit
    ->   6: load_var n
    ->   7: push_const 1.0
    ->   8: sub 2
    ->   9: load_var fib
    ->  10: call_lambda 1
    ->  11: load_var n
    ->  12: push_const 2.0
    ->  13: sub 2
    ->  14: load_var fib
    ->  15: call_lambda 1
    ->  16: add 2
   1: store_var fib
   2: push_const 10.0
   3: load_var fib
   4: call_lambda 1
   5: load_var print
   6: call_lambda 1
```

<br>

Compile to JavaScript:

```
./program --js < fib10.lisp
/* lisp-to-js */
let print = console.log;


(() => {
let fib =  ((n) =>  n  < 2 ?  n  : ( fib (( n -1), )+ fib (( n -2), ))

)
; print ( fib (10, ), )
})()
```

<br>

The implemented optimizations are constant folding and propagation, and dead
code elimination:

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

### Run

Required (one of):

- `--js` output JavaScript to stdout
- `--vm` compile to bytecode and execute in VM

Optional:

- `--optimize` for optimization
- `--debug` show annotated bytecode

<br>

### Tests

```
cargo test
```

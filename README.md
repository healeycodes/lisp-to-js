# lisp-to-js

> My blog posts:
>
> - [Compiling Lisp to Bytecode and Running It](https://healeycodes.com/compiling-lisp-to-bytecode-and-running-it)
> - [Lisp Compiler Optimizations](https://healeycodes.com/lisp-compiler-optimizations)
> - [Lisp to JavaScript Compiler](https://healeycodes.com/lisp-to-javascript-compiler)

<br>

This project is an optmizing Lisp compiler and bytecode VM.

It can compile to JavaScript, or compile to bytecode and execute in a VM.

The bytecode VM is a little faster than Node.js (TODO: benchmarks).

<br>

Bytecode VM:

```
./program --vm --debug < fib10.lisp
   0: PushClosure(["n"])
    ->   0: LoadVar("n")
    ->   1: PushConst(2.0)
    ->   2: LessThan
    ->   3: Jump(6) // go to 6
    ->   4: LoadVar("n")
    ->   5: Jump(17) // exit
    ->   6: LoadVar("n")
    ->   7: PushConst(1.0)
    ->   8: Sub(2)
    ->   9: LoadVar("fib")
    ->  10: CallLambda(1)
    ->  11: LoadVar("n")
    ->  12: PushConst(2.0)
    ->  13: Sub(2)
    ->  14: LoadVar("fib")
    ->  15: CallLambda(1)
    ->  16: Add(2)
   1: StoreVar("fib")
   2: PushConst(10.0)
   3: LoadVar("fib")
   4: CallLambda(1)
   5: LoadVar("print")
   6: CallLambda(1)

55
```

<br>

Compile to JavaScript:

```
./program --js < fib10.lisp
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

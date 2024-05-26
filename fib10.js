/* lisp-to-js */
let print = console.log;


let fib =  ((n) =>  n  < 2 ?  n  : ( fib (( n -1), )+ fib (( n -2), ))

)
; print ( fib (10, ), )

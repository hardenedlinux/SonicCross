(define-module (scripts torchgen)
  #:use-module (ice-9 getopt-long)
  #:export (main))

(define %summary "SonicCross torchgen bootstrap command.")
(define %synopsis "torchgen [--help]")
(define %help "The real torchgen CLI will be added after project infrastructure is stable.\n")

(define (usage)
  (display "SonicCross torchgen bootstrap command\n")
  (display "Usage: guild torchgen [--help]\n"))

(define (main . args)
  (let ((options (getopt-long args '((help (single-char #\h))))))
    (if (option-ref options 'help #f)
        (usage)
        (usage)))
  0)

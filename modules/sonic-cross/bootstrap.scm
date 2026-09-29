(define-module (sonic-cross bootstrap)
  #:export (sonic-cross-bootstrap-message))

(define (sonic-cross-bootstrap-message)
  "Return a marker proving that the SonicCross module namespace loads."
  'sonic-cross-bootstrap)

;;  -*-  indent-tabs-mode:nil; coding: utf-8 -*-
;;  Copyright (C) 2026
;;      "Mu Lei" known as "NalaGinrut" <roy@hardenedlinux.org>
;;  SonicCross is free software: you can redistribute it and/or modify
;;  it under the terms of the GNU General Public License published
;;  by the Free Software Foundation, either version 3 of the License,
;;  or (at your option) any later version.

;;  SonicCross is distributed in the hope that it will be useful,
;;  but WITHOUT ANY WARRANTY; without even the implied warranty of
;;  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
;;  GNU General Public License for more details.

;;  You should have received a copy of the GNU General Public License
;;  along with this program. If not, see <http://www.gnu.org/licenses/>.

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

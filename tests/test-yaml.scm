#!/usr/bin/guile --no-auto-compile
!#

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

(use-modules (srfi srfi-64)
             (srfi srfi-1)
             (sonic-cross yaml))

(define (scalar-value node)
  (yaml-scalar-value node))

(define (mapping-value mapping key)
  (cadr (find (lambda (entry)
                (and (yaml-scalar? (car entry))
                     (string=? (yaml-scalar-value (car entry)) key)))
              (yaml-mapping-entries mapping))))

(test-begin "sonic-cross yaml")

(let ((root (yaml-load-string "foo: bar\n")))
  (test-assert (yaml-mapping? root))
  (test-equal "bar" (scalar-value (mapping-value root "foo"))))

(let ((root (yaml-load-string "foo:\n  bar: baz\n")))
  (test-equal "baz"
    (scalar-value (mapping-value (mapping-value root "foo") "bar"))))

(let* ((root (yaml-load-string "items:\n  - one\n  - two\n"))
       (items (mapping-value root "items")))
  (test-assert (yaml-sequence? items))
  (test-equal '("one" "two")
    (map yaml-scalar-value (yaml-sequence-items items))))

(let* ((root (yaml-load-string
              "root:\n  list:\n    - name: foo\n      value: bar\n")))
  (test-equal "foo"
    (scalar-value
     (mapping-value (car (yaml-sequence-items
                         (mapping-value (mapping-value root "root") "list")))
                    "name"))))

(let ((root (yaml-load-string "empty_map: {}\nempty_list: []\n")))
  (test-assert (yaml-mapping? (mapping-value root "empty_map")))
  (test-assert (null? (yaml-mapping-entries (mapping-value root "empty_map"))))
  (test-assert (yaml-sequence? (mapping-value root "empty_list")))
  (test-assert (null? (yaml-sequence-items (mapping-value root "empty_list")))))

(let ((root (yaml-load-string "a: \"123\"\nb: 'true'\n")))
  (test-assert (yaml-scalar? (mapping-value root "a")))
  (test-equal "123" (scalar-value (mapping-value root "a")))
  (test-equal 'double-quoted (yaml-scalar-style (mapping-value root "a")))
  (test-equal "true" (scalar-value (mapping-value root "b")))
  (test-equal 'single-quoted (yaml-scalar-style (mapping-value root "b"))))

(let ((root (yaml-load-string
             "literal: |\n  hello\n  world\nfolded: >\n  hello\n  world\n")))
  (test-equal "hello\nworld\n" (scalar-value (mapping-value root "literal")))
  (test-equal "hello world\n" (scalar-value (mapping-value root "folded")))
  (test-equal 'literal (yaml-scalar-style (mapping-value root "literal")))
  (test-equal 'folded (yaml-scalar-style (mapping-value root "folded"))))

(let ((root (yaml-load-string
             "truth: true\nnumber: 123\nnull_value: null\n")))
  (test-equal "true" (scalar-value (mapping-value root "truth")))
  (test-equal "123" (scalar-value (mapping-value root "number")))
  (test-assert (yaml-null? (mapping-value root "null_value"))))

(let ((root (yaml-load-string "defaults: &defaults\n  value: base\ncopy: *defaults\n")))
  (test-equal "defaults" (yaml-mapping-anchor (mapping-value root "defaults")))
  (test-assert (yaml-alias? (mapping-value root "copy")))
  (test-equal "defaults" (yaml-alias-anchor (mapping-value root "copy"))))

(let ((root (yaml-load "fixtures/native-functions.yaml")))
  (test-assert (yaml-sequence? root))
  (test-equal 2 (length (yaml-sequence-items root))))

(test-error 'yaml-error (yaml-load "fixtures/does-not-exist.yaml"))
(test-error 'yaml-error (yaml-load-string "broken: [one\n"))

(let ((runner (test-runner-current)))
  (test-end)
  (exit (if (zero? (test-runner-fail-count runner)) 0 1)))

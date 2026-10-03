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

;; Faithful port of torchgen.code_template.CodeTemplate.substitute, which is
;; the template engine used to render every generated C++ artifact.  Semantics:
;;
;;   - `$name` / `${name}` / `${,name}` / `${name,}` are substitutions.
;;   - If the placeholder starts a line (only whitespace precedes it since the
;;     previous newline), the value is block-substituted: each element (or the
;;     single scalar, coerced to a list) is placed on its own line, indented to
;;     the placeholder's column, trailing whitespace stripped.
;;   - Otherwise the value is inline: a list is joined with ", " (with an
;;     optional leading/trailing comma per `${,name}` / `${name,}`), a scalar is
;;     stringified.

(define-module (sonic-cross code-template)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:export (code-template-substitute))

(define (stringify value)
  (cond ((string? value) value)
        ((number? value) (number->string value))
        ((boolean? value) (if value "#t" "#f"))
        ((symbol? value) (symbol->string value))
        (else (error 'stringify value))))

(define (char-whitespace-only? line)
  (string-every char-whitespace? line))

(define (py-splitlines s)
  ;; Python str.splitlines() over content whose only line terminator is #\newline:
  ;; a trailing newline is a terminator, not a separator, so "a\n" splits to
  ;; ("a") rather than ("a" "").  This differs from string-split, which keeps a
  ;; trailing empty element and would inject spurious blank lines.
  (let ((parts (string-split s #\newline)))
    (if (and (pair? parts) (string-null? (car (last-pair parts))))
        (drop-right parts 1)
        parts)))

(define (textwrap-indent text prefix)
  ;; textwrap.indent(text, prefix) with its default predicate line.strip():
  ;; prefix each non-blank line, preserving line terminators via
  ;; str.splitlines(keepends=True).  `text` from block-substitute step 1 never
  ;; carries a trailing newline, but keepends is honoured for faithfulness.
  (define (with-keepends)
    (let ((parts (py-splitlines text)))
      (cond
       ((null? parts) '())
       ((char=? (string-ref text (1- (string-length text))) #\newline)
        (map (lambda (p) (string-append p "\n")) parts))
       (else
        (append (map (lambda (p) (string-append p "\n"))
                     (drop-right parts 1))
                (list (car (last-pair parts))))))))
  (string-concatenate
   (map (lambda (line)
          (if (char-whitespace-only? line) line (string-append prefix line)))
        (if (string-null? text) '() (with-keepends)))))

(define (block-substitute indent values)
  ;; values :: (listof string); mirror CodeTemplate.indent_lines exactly:
  ;;   1. content = "\n".join(chain(str(e).splitlines() for e in values))
  ;;   2. content = textwrap.indent(content, prefix=indent)
  ;;   3. "\n".join(map(str.rstrip, content.splitlines())).rstrip()
  (let* ((content (string-join (append-map py-splitlines values) "\n"))
         (indented (textwrap-indent content indent))
         (rstripped (string-join
                     (map string-trim-right (py-splitlines indented))
                     "\n")))
    (string-trim-right rstripped)))

(define (inline-substitute value comma-before comma-after)
  (if (list? value)
      (let ((middle (string-join (map stringify value) ", ")))
        (if (null? value)
            middle
            (string-append comma-before middle comma-after)))
      (stringify value)))

;; Parse the placeholder after a '$'.  Returns (values key comma-before
;; comma-after consumed) where `consumed` is the number of characters consumed
;; after the '$' (i.e. the placeholder length).
(define (parse-placeholder pattern start)
  ;; start is the index just after the '$'.
  (let* ((len (string-length pattern)))
    (if (and (< start len) (char=? (string-ref pattern start) #\{))
        (let ((close (string-index pattern #\} (1+ start))))
          (unless close (error 'code-template "unclosed { in template"))
          (let* ((inner (substring pattern (1+ start) close))
                 (before? (string-prefix? "," inner))
                 (after? (string-suffix? "," inner))
                 (key (substring inner (if before? 1 0)
                                 (- (string-length inner) (if after? 1 0)))))
            (values key (if before? ", " "") (if after? ", " "")
                    (+ (- close start) 1))))
        ;; bare identifier: [^\d\W]\w*  => first char letter/underscore, rest word.
        (let loop ((end start))
          (cond
           ((>= end len) (values (substring pattern start end) "" "" (- end start)))
           ((char-set-contains? char-set:letter+digit
                                (string-ref pattern end))
            (loop (1+ end)))
           ((char=? (string-ref pattern end) #\_)
            (loop (1+ end)))
           (else (values (substring pattern start end) "" "" (- end start))))))))

(define (line-start-indent pattern dollar)
  ;; If the '$' at `dollar` is preceded only by whitespace since the previous
  ;; newline (or start of string), return that whitespace; otherwise #f.
  (let loop ((j (1- dollar)) (ws '()))
    (if (< j 0)
        (list->string (reverse ws))
        (let ((c (string-ref pattern j)))
          (cond
           ((char=? c #\newline) (list->string (reverse ws)))
           ((char-whitespace? c) (loop (1- j) (cons c ws)))
           (else #f))))))

(define (code-template-substitute pattern lookup)
  (let ((len (string-length pattern)))
    (let loop ((i 0) (chunks '()))
      (let ((dollar (string-index pattern #\$ i)))
        (if (not dollar)
            (string-concatenate-reverse (cons (substring pattern i) chunks))
            (call-with-values
                (lambda () (parse-placeholder pattern (1+ dollar)))
              (lambda (key comma-before comma-after consumed)
                (let* ((indent (line-start-indent pattern dollar))
                       (value (lookup key)))
                  (unless value
                    (error 'code-template "unknown key" key))
                  (let* ((replacement
                          (if indent
                              (block-substitute
                               indent
                               (if (list? value) value (list (stringify value))))
                              (inline-substitute value comma-before comma-after)))
                         ;; The regex match spans the leading whitespace (the
                         ;; `indent` group) through the placeholder; re.sub
                         ;; replaces that whole span.  So when `indent` is set,
                         ;; the literal chunk before the match must exclude it.
                         (match-start (if indent
                                          (- dollar (string-length indent))
                                          dollar)))
                    (loop (+ (1+ dollar) consumed)
                          (cons* replacement (substring pattern i match-start)
                                 chunks)))))))))))

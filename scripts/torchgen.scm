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

;; The `guild torchgen` command — SonicCross's production entry point, a drop-in
;; replacement for `python3 torchgen/gen.py`.  It turns native_functions.yaml
;; into the full C++ artifact closure written under --install-dir, using only
;; Guile/Scheme and the libyaml FFI (no Python anywhere in this path).

(define-module (scripts torchgen)
  #:use-module (ice-9 getopt-long)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-13)
  #:use-module (sonic-cross generate)
  #:export (main))

(define %summary "Generate ATen source files.")
(define %synopsis "torchgen [-s SOURCE] [-d DIR] [-o DEPFILE] [--dry-run] [--generate SUBSETS]")
(define %help "\
Generate the ATen C++ source/header artifacts from native_functions.yaml.

  -s, --source-path PATH    source directory for ATen (default: aten/src/ATen)
  -d, --install-dir DIR     output directory (default: build/aten/src/ATen)
  -o, --output-dependencies FILE
                            write a list of generated files into FILE and exit
      --dry-run             run without writing any files
      --generate SUBSETS    comma-separated subset of headers,sources,
                            declarations_yaml (default: all three)
      --aoti-install-dir DIR   AOTInductor shim output directory (unused)
      --headeronly-install-dir DIR  header-only output directory
      -h, --help            show this help
")

(define (usage)
  (display "SonicCross torchgen command\n")
  (display "Usage: guild torchgen [options]\n\n")
  (display %help))

;; ---------------------------------------------------------------------------
;; argument parsing
;; ---------------------------------------------------------------------------

(define %option-spec
  '((help (single-char #\h))
    (source-path (single-char #\s) (value #t))
    (install-dir (single-char #\d) (value #t))
    (output-dependencies (single-char #\o) (value #t))
    (dry-run)
    (generate (value #t))
    (aoti-install-dir (value #t))
    (headeronly-install-dir (value #t))))

(define (parse-subsets str)
  ;; "headers,sources" -> (headers sources)
  (map string->symbol
       (remove string-null? (string-split str #\,))))

;; ---------------------------------------------------------------------------
;; --output-dependencies: torchgen FileManager.write_outputs
;; ---------------------------------------------------------------------------

(define (last-path-component p)
  (car (last-pair (string-split p #\/))))

(define (parent-path p)
  (let ((parts (string-split p #\/)))
    (if (null? (cdr parts))
        "."
        (string-join (drop-right parts 1) "/"))))

(define (strip-extension name)
  (let ((idx (string-index-right name #\.)))
    (if idx (substring name 0 idx) name)))

(define (everything-shard? filename)
  ;; torchgen's write_sharded_with_template discards the monolithic
  ;; "FooEverything.cpp" from FileManager.files (they are not meant to be
  ;; compiled), so they must be omitted from the --output-dependencies depfile.
  (string-suffix? "Everything" (strip-extension filename)))

(define (write-depfile path varname paths)
  ;; Mirrors FileManager.write_outputs:
  ;;   set(
  ;;   <varname>
  ;;       "<path>"
  ;;       ...
  ;;   )
  (call-with-output-file path
    (lambda (port)
      (display "set(\n" port)
      (display varname port)
      (newline port)
      (for-each (lambda (p)
                  (display "    \"" port)
                  (display p port)
                  (display "\"\n" port))
                paths)
      (display ")" port))))

(define (write-output-dependencies depfile-path written)
  ;; written :: ((label dir filename) ...).  torchgen emits one depfile per file
  ;; manager: cpu -> <name>, cpu_vec -> cpu_vec_<name>, core -> core_<name>,
  ;; ops -> ops_<name>, cuda -> cuda_<name> (headeronly/aoti are not listed).
  (let* ((name (last-path-component depfile-path))
         (parent (parent-path depfile-path))
         (stem (strip-extension name)))
    (for-each
     (lambda (group)
       (let* ((label (car group))
              (prefix (cdr group))
              (paths (sort
                      (map (lambda (w) (string-append (cadr w) "/" (caddr w)))
                           (filter (lambda (w)
                                     (and (eq? (car w) label)
                                          (not (everything-shard? (caddr w)))))
                                   written))
                      string<?)))
         (write-depfile (string-append parent "/" prefix name)
                        (string-append prefix stem)
                        paths)))
     '((cpu . "") (cpu-vec . "cpu_vec_") (core . "core_")
       (ops . "ops_") (cuda . "cuda_")))))

;; ---------------------------------------------------------------------------
;; main
;; ---------------------------------------------------------------------------

(define (main . args)
  ;; guild invokes (main arg1 arg2 ...) with no program name, but getopt-long
  ;; requires the program name as the first element (it skips it).  Prepend a
  ;; dummy name so `--help`/`-s`/... are parsed rather than treated as the
  ;; program name and silently dropped.
  (let ((options (getopt-long (cons "torchgen" args) %option-spec)))
    (cond
     ((option-ref options 'help #f)
      (usage)
      0)
     (else
      (let* ((source-path (option-ref options 'source-path "aten/src/ATen"))
             (install-dir (option-ref options 'install-dir "build/aten/src/ATen"))
             (output-deps (option-ref options 'output-dependencies #f))
             (dry-run? (option-ref options 'dry-run #f))
             (aoti-dir (option-ref options 'aoti-install-dir #f))
             (headeronly-dir (option-ref options 'headeronly-install-dir #f))
             (subsets (let ((g (option-ref options 'generate #f)))
                        (if g
                            (parse-subsets g)
                            '(headers sources declarations_yaml)))))
        (let ((yaml-path (string-append source-path "/native/native_functions.yaml")))
          (if (not (file-exists? yaml-path))
              (begin
                (format (current-error-port)
                        "torchgen: ~a: no such file~%" yaml-path)
                (format (current-error-port)
                        "  (point -s/--source-path at an ATen source tree)~%")
                (newline (current-error-port))
                (usage)
                1)
              (let ((written (generate-all source-path install-dir
                                           #:aoti-install-dir aoti-dir
                                           #:headeronly-install-dir headeronly-dir
                                           #:dry-run? dry-run?
                                           #:generate subsets)))
                (when output-deps
                  (write-output-dependencies output-deps written))
                0))))))))

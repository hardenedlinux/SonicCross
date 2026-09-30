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
             (sonic-cross irregex))

(test-begin "irregex integration")

(let* ((regexp (string->irregex "([^\\(]+)\\((.*)\\) -> (.*)"))
       (match (irregex-match regexp "add.Tensor(Tensor self) -> Tensor")))
  (test-assert "matches schema declaration" match)
  (test-equal "operator capture"
    "add.Tensor"
    (irregex-match-substring match 1))
  (test-equal "arguments capture"
    "Tensor self"
    (irregex-match-substring match 2))
  (test-equal "returns capture"
    "Tensor"
    (irregex-match-substring match 3)))

(test-end "irregex integration")

#!/bin/bash

export HTTP_PROXY="http://127.0.0.1:1082"
export HTTPS_PROXY="http://127.0.0.1:1082"
export WSS_PROXY="http://127.0.0.1:1082"
export ALL_PROXY="http://127.0.0.1:1082"
export NODE_EXTRA_CA_CERTS="/etc/ssl/certs/ca-certificates.crt"
export NODE_TLS_REJECT_UNAUTHORIZED=0
codex exec "hello"


#!/bin/bash
find  ~/WSL-projs/experiments/AdaSparseChem-cb29-10task -type f -name "*.tar"  -mmin +$1 -fprintf del_tar_files.sh 'rm -rf %P \n'
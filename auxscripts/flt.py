#!/usr/bin/env python3

import librosa
import sys

from os import listdir
from os.path import isfile, join

duration_setting = {
    "min": 1.0,
    "max": 30.0,
}

if __name__ == "__main__":
    sr = 22050
    tot = 0
    totflt = 0
    cnt = 0
    mypath=sys.argv[1]
    for fraw in listdir(mypath):
        f = join(mypath, fraw)
        if isfile(f):
            try:
                speech, orig_sr = librosa.load(f, sr=sr)
                
                len_speech = len(speech)
                flt = "remove" if len_speech < sr or len_speech > sr * 30 else "keep"
                print(flt, f)
                if not cnt % 100:
                    print(f"processed {cnt} files", file=sys.stderr)
                tot += len_speech
                if flt == "keep":
                    totflt += len_speech
                    cnt += 1
            except Exception as e:
                pass
    print(f"total sec: {tot/sr}, after filtering: {totflt/sr}, avg segm len in sec (flt): {totflt/(sr*cnt)}", file=sys.stderr)

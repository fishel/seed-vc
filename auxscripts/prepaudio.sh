#!/bin/bash

dd=$1

ffmpeg="/scratch/project_12345/code/seed-vc/ffmpeg-7.0.2-amd64-static/ffmpeg"

algo="silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,areverse,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB,areverse,loudnorm=I=-16:TP=-1.5:LRA=11"

l=$( ls $dd/* | wc -l )

i=1
for f in $dd/*
do
	outf=$( echo $f | sed -e "s/rawlonger/clnlonger/g;s/mp3/wav/g" )
	
	if [[ -e "$outf" ]]
	then
		echo "$i / $l:" $outf already exists
	else
		$ffmpeg -hide_banner -loglevel error -i "$f" -af "$algo" -ar 22050 -ac 1 $outf
		echo "$i / $l:" $f "-->" $outf
	fi
	i=$[ $i + 1 ]
done

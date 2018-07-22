#!/bin/bash

MODEL=ssd_mobilenet_v1_coco_11_06_2017

ARCHIVE=$MODEL.tar.gz

rm -f $FILE
rm -f $ARCHIVE
wget http://download.tensorflow.org/models/object_detection/$ARCHIVE
if [ -f $ARCHIVE ]; then
    tar xfvz $ARCHIVE
    mv $MODEL/*.ckpt* .
    rm -r $MODEL
    rm $ARCHIVE
else
    echo "Download of $FILE failed"
fi
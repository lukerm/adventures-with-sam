sudo apt install imagemagick

IMG_DIR=$HOME/adventures-with-sam/data/img/xmas
mkdir -p $IMG_DIR/small

for f in `ls $IMG_DIR/*.jpg`; do
    # check file size > 1MB
    size=$(stat -c %s $f)
    if [ $size -gt 1000000 ]; then
        echo "Resizing $f"
    else
        echo "Skipping $f"
        continue
    fi
    # Replace extension to .small.jpg
    convert $f -resize 30% ${IMG_DIR}/small/`echo $(basename $f) | sed -E 's/\.jpg/\.small\.jpg/g'`
done

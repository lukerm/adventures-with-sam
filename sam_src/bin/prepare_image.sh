#  Copyright (C) 2024 lukerm of www.zl-labs.tech
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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


#!/usr/bin/env bash
set -e

### Find wheel packages, updates with the desired version, then delete the old wheels.
### Local Test:

# Download test wheels.
#wget -P test/path_1 https://github.com/tenstorrent/tt-forge/releases/download/nightly-2025-03-25T06-02-03/forge-0.1.250324+dev.39df478d-cp310-cp310-linux_x86_64.whl
#wget -P test/path_2 https://github.com/tenstorrent/tt-forge/releases/download/nightly-2025-03-25T06-02-03/tt_torch-0.1-cp311-cp311-linux_x86_64.whl
#wget -P test/path_2 https://github.com/tenstorrent/tt-forge/releases/download/nightly-2025-03-25T06-02-03/tvm-0.14.0+dev.tt.631209194-cp310-cp310-linux_x86_64.whl

# Set required ENVS
#WHEEL_VERSION='20250326172139.nightly'
#WHEEL_ROOT_PATH='test'

#### Install wheel via pip
python3 -m venv venv
source venv/bin/activate
pip install wheel

# Ref for vaild python versions
# https://packaging.python.org/en/latest/discussions/versioning/#valid-version-numbers
# https://calver.org/
if [[ -z "$WHEEL_VERSION" ]]; then
    echo "ENV WHEEL_VERSION is not set. exiting.."
    exit 1
fi

if [[ -z "$WHEEL_ROOT_PATH" ]]; then
    echo "ENV WHEEL_ROOT_PATH is not set. exiting.."
    exit 1
fi
# Get all the directories that contain wheel files
wheel_paths=$(find $WHEEL_ROOT_PATH -type f -iname "*.whl"  -exec dirname {} \; | uniq | xargs)

for wheel_path in $wheel_paths; do

    pushd $wheel_path

    # Exploded all wheel packages
    ls *.whl | xargs -I{} wheel unpack {}
    # Delete old wheel
    rm *.whl

    # Update the METADATA file with new version.
    perl  -X -p -i -e  's!(?<=^Version:\s)[\w\.\+]+!'$WHEEL_VERSION'!g' */*/METADATA

    root_folders=$(find . -maxdepth 1 -type d -printf '%f\n' | grep -P '(?<=\-)[\w\.\+]+' | xargs )

    # Rename folders root folder to match new version
    for rf in $root_folders; do
        package_prefix="$(echo $rf| grep -oP '^\w+-')"
        new_folder_name="$(echo $rf| perl -X -p -e 's!(?<=\-)[\w\.\+]+!'$WHEEL_VERSION'!g')"
        mv $rf $new_folder_name
        pushd $new_folder_name
        ls
        child_folders=$(find . -maxdepth 1 -type d -printf '%f\n' | grep -P '(?<=\-)[\w\.\+]+(?=\.dist-info|\.data)' | xargs )
        # Rename folders with .dist-info and/or .data to match new version
        for cf in $child_folders; do
            new_folder_name="$(echo $cf| perl -X -p -e 's!(?<=\-)[\w\.\+]+(?=\.dist-info|\.data)!'$WHEEL_VERSION'!g')"
            mv $cf $new_folder_name
        done
        popd
        ls
        # Repack wheel
        ls | grep "$package_prefix$WHEEL_VERSION" | xargs -I{} wheel pack {}
        # Delete old exploded wheel folder
        ls -d $package_prefix*$WHEEL_VERSION | xargs -I{} rm -r {}
        ls
    done

    popd

done

#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Are we on main branch
ON_MAIN=$(git branch --show-current | grep -q main && echo "true" || echo "false")

build() {

    echo "Building image $IMAGE_NAME:$DOCKER_TAG"
    docker build \
        --progress=plain \
        --build-arg FROM_TAG=$DOCKER_TAG \
        ${FROM_IMAGE:+--build-arg FROM_IMAGE=$FROM_IMAGE} \
        -t $IMAGE_NAME:$DOCKER_TAG \
        -t $IMAGE_NAME:latest \
        -f $DOCKERFILE .

}

build_and_push() {

    if [ "$FORCE_BUILD" = "true" ]; then
        build $IMAGE_NAME $DOCKERFILE $FROM_IMAGE
    else
        # Check if image already exists
        set +e
        if docker manifest inspect $IMAGE_NAME:$DOCKER_TAG > /dev/null; then
            set -e
            echo "Image $IMAGE_NAME:$DOCKER_TAG already exists"
            SKIP_PUSH="true"
        else
            build $IMAGE_NAME $DOCKERFILE $FROM_IMAGE
        fi
    fi

    # Push image
    if [ "$SKIP_PUSH" = "false" ]; then
        echo "Pushing image $IMAGE_NAME:$DOCKER_TAG"
        docker push $IMAGE_NAME:$DOCKER_TAG
    fi

    # If we are on main branch we always push latest tag even if image already exists or "$SKIP_PUSH" = "true"
    if [ "$ON_MAIN" = "true" ]; then
        printf "\nPushing latest tag for $IMAGE_NAME"
        # Used to push both tags in one command on the original sha256 sum layer remotely (docker create manifest create a new sha256 sum only with only with new tag)
        docker buildx imagetools create $IMAGE_NAME:$DOCKER_TAG --tag $IMAGE_NAME:latest --tag $IMAGE_NAME:$DOCKER_TAG
    fi
}

build_and_push $IMAGE_NAME $DOCKERFILE $FROM_IMAGE

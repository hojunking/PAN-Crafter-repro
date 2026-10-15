# RepoDigest verified to resolve to the existing local base image/config ID
# sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887.
# A config/image ID is not a registry manifest digest and cannot be used in FROM.
FROM hojunqueen/pancrafter-env@sha256:cdee3ba6a544f465c72e86af981dd6924cd07bb5de24bebc6682c30f03c177b9

LABEL org.panpv.base_image_id="sha256:ebe266ad6514c1602b423518f77bf87e57e9f6e51cca9d2ac21581a64d106887"
LABEL org.panpv.purpose="Isolated PANPV GDAL WeightedBrovey8 runtime; no host package changes"

USER root
# Ubuntu's GDAL requires GLIBCXX_3.4.30, newer than the base Conda runtime.
# Preload only this derived image's backwards-compatible system C++ runtime.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libgdal-dev g++ \
    && panpv_gdal_version="$(gdal-config --version)" \
    && python -m pip install --no-cache-dir --no-build-isolation "GDAL==${panpv_gdal_version}"
# Conda also carries an older same-soname TIFF. Resolve GDAL against the
# installed Ubuntu TIFF, not that incompatible copy (TIFFReadRGBATileExt).
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libtiff.so.5
RUN python -c "import numpy, torch; from osgeo import gdal; assert hasattr(gdal, 'CreatePansharpenedVRT'); print({'gdal':gdal.VersionInfo('RELEASE_NAME'),'numpy':numpy.__version__,'torch':torch.__version__})"

ENV GDAL_NUM_THREADS=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace

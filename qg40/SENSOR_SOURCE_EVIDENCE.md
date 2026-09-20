# QG40 QB/GF2 local source binding evidence — 2026-09-20

This record binds the existing local corpus; it is not a new dataset or a claim
that all preflight checks have already passed. `sensor_sources.json` pins the
actual H5 bytes. Every server must resolve its own relative paths, verify those
hashes, and execute the source/LP/QB-repair checks before LOCAL_READY. A missing
or different source must not be downloaded, generated, or silently substituted.

## Channel order and units: source-repository convention

The declared training order is **B, G, R, NIR**, with RGB display indices
**2, 1, 0**. This is the convention documented for the installed PanCollection
QB/GF2 corpus, not a spectral measurement or an embedded H5 band-name attribute.
The train/validation/RR source paths resolve directly to the CANConv corpus
described by the following local evidence:

- `CANConv/results_log/2026-08-10_pancollection-dataset.md`, lines 64–70:
  explicitly identifies the four QB/GF2 channels as Blue, Green, Red, NIR and
  attributes that convention to the CANConv RGB helper and DLPan MTF source.
  SHA256 `f18b588f71f893bb0d7b48444dfbac90358d05f31ad02fb8a0a0ae4a851a3afc`.
- `CANConv/canconv/util/log.py`, lines 82–83: four-channel display selects
  `[2, 1, 0]`. SHA256
  `ca4943ade70376228d986aa5d2d9b2ba7617f59c2dfdb3704e1e7bd8982f3d53`.
- `CANConv/canconv/dataset/h5pan.py`: converts stored arrays to float32 and
  divides QB by 2047 and GF2 by 1023; it does not permute spectral channels.
  SHA256 `56d86d661afd99bffbde2c968ffa82234c5cbb064453f9e3edf514a191e5e05a`.
- `DLPan-Toolbox/02-Test-toolbox-for-traditional-and-DL(Matlab)/Tools/genMTF.m`,
  line 31: QB gains `[0.34, 0.32, 0.30, 0.22]` explicitly state B,G,R,NIR order.
  GF2 has no dedicated branch and uses the documented default 0.3 per band.
  SHA256 `5c96781d777ec6f08fee696ff80dc46972875b764c6f289388bb21614dd14c2d`.

These files were read under `/home/knuvi/Desktop/song/` on the source machine.
This record preserves their relevant observations and hashes for handoff; other
servers need not have that absolute directory layout. GF2 default gains are
the evaluation protocol, not a certification of the sensor's physical MTF.

Actual H5 root and dataset attributes were inspected for all ten selected/raw
files. None contains a band-name/order attribute. Raw train/val/RR have no
attributes; QB repaired files contain the repair recipe, raw filename, and
2026-09-08 build timestamp. The FR files contain a PanCollection FullData MAT
source-folder attribution. First-sample values are in DN-scale hundreds or
thousands, not [0,1] or [-1,1]. Full finite/range/source checks remain preflight
work; the nominal maxDN constants are not fitted from observed maxima.

## Exact files and split correspondence

- QB train/val: existing `train_qb_msfix.h5` / `valid_qb_msfix.h5`, counts
  17139/1905. Raw originals are the corresponding `train_qb.h5` / `valid_qb.h5`
  links to the CANConv dataset. The recorded repair is sensor-QB MTF filtering
  with replicate boundaries, HR `[2::4,2::4]` decimation, and `interp23tap` LMS.
  Preflight must prove **every** raw/fixed GT/PAN pair unchanged and MS/LMS
  recipe correspondence; the attribute alone is not a PASS.
- GF2 train/val: unchanged CANConv-installed `train_gf2.h5` /
  `valid_gf2.h5`, counts 19809/2201.
- RR: the existing sensor-specific `test_*_multiExm1.h5`, 20 samples,
  GT/MS/LMS/PAN stored jointly in each file. Original row order is retained;
  no spatial or spectral test transformation is introduced.
- FR: the sensor's **20 MAT scenes**, numerically ordered fr1 through fr20,
  not a substitute OrigScale H5 scene set. On 2026-09-20, all 20 MAT SHA256
  values were checked against each `full_examples_mat20/provenance.json` and
  all MS/LMS/PAN arrays were compared to their H5 rows: **exact equality for
  both QB and GF2** after HWC→CHW only. No FR reconstruction GT is assumed.

The FR source identifier is the H5's existing attribution to PanCollection
Testing Dataset (FullData, MAT), Drive folder
`16pGIqvwWfyQVvkk3s1xrwLpavqQd0Bv7`; this audit did not access Drive.
The source manifest hashes are:

| File | SHA256 |
|---|---|
| QB `full_examples_mat20/provenance.json` | `47d592ef4a120d5aa2956817adf2916d93c8576412e9aa18bf3632b24dbe8c65` |
| GF2 `full_examples_mat20/provenance.json` | `0aaacb2ed0fe99b9bc24409b5e78ca7af3c1ed82c5e3c6c497eb1bfdd1fd7900` |
| `tools/build_fr_paperset.py` | `a514ad316328dbfe17096451196610d9860b393b4b79adfd6580c3ac60b7e116` |
| `tools/repair_qb_ms.py` | `0ed95018fd88afde7194eac8fdd5ae79de7b9056574766e339726dbc0a7c8884` |

These checks establish source identity and within-file array correspondence,
not geospatial disjointness of all train/validation/test scenes. No claim that
RR scene i and FR scene i are the same geographic scene is made.

## No-data and saturation policy

No explicit no-data sentinel or mask was found in the inspected H5 metadata.
That absence is not evidence that every zero is a missing pixel or that no
missing observations exist. Preserve all finite stored values, including zeros;
count them diagnostically and do not drop samples, mask borders, or fit a new
normalization. Reject nonfinite source values during P0 scanning.

Preserve values equal to nominal maxDN and report saturation counts. Do not
clip or repair training inputs or GT. P0 requires PAN/GT within the declared
physical DN interval; finite MS/LMS interpolation/filter ringing outside it
is retained and reported (it exists in QB LMS). Prediction clipping remains
the unchanged official evaluator policy, not an input transformation.

The LP feature cache is separately generated by the already fixed QG40 recipe
from each verified source PAN; it is not the sensor MTF and is not trusted on
filename alone. The historical repair script's LR1px/HR4px wording is not
copied as physical displacement evidence: the independent phase audit verifies
that one HR decimation-phase index is one HR pixel, or 0.25 LR spacing.

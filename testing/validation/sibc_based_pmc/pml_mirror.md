# PMC walls through longitudinal PML

The public `SurfaceImpedance(id="wall", resistance=float("inf"))` boundary
now extends through a PML whose absorption direction is tangent to the wall.
The wall and retained material must continue uniformly through the slab and
its neighbouring field stencils. These tests use native PML at both ends.

For a retained electric dual area A and longitudinal signed derivative D,
the boundary circulation contains A D. At a flat PMC face, both the
longitudinal H path and the electric area are halved, so their factors cancel.
The existing PML convolution therefore operates on exactly the ordinary D.
Its correction Q contributes an additional A Q to the boundary solve.
The normal-to-wall circulation retains its clipped geometry, including the
factor of two at a flat face.

The implementation captures Q with the existing electric PML kernels and
their normal histories, restores the original electric field, then applies
the additional circulation to the sparse solve. This ordering also preserves
the old electric field needed by finite-impedance implicit solves and ADE
states. PMC is its exact zero-admittance special case.

## Independent image comparison

`pml_mirror.py` compares two independently built 3D grids: a full ordinary
grid, and a grid whose upper half is excluded by the exact SIBC PMC.
The full grid has no impedance material or sparse boundary updates.
Random initial fields obey the image parity: electric normal and magnetic
tangential components are odd; electric tangential and magnetic normal
components are even. Both grids use identical PML coefficient profiles.

All retained components of E and H, and all retained electric and magnetic
PML histories, are compared after every step. The 24 cases cover all three
absorption axes, both PML end directions simultaneously, HORIPML and MRIPML,
orders one and two, and both CPU precisions.

Results over 200 steps, saved in [pml_mirror.json](results/pml_mirror.json):

| Precision | Maximum field error / initial field norm | Maximum PML history relative error |
| --- | ---: | ---: |
| Double | 4.38e-15 | 1.01e-14 |
| Single | 3.10e-6 | 1.08e-5 |

All 24 cases passed. A negative control disables the captured boundary PML
forcing while retaining the PML histories: its maximum field error is 8.865
and its maximum history relative error is 68.69. Merely permitting geometry
overlap without the boundary PML forcing fails this comparison.

These image tests verify exact PMC coupling over the tested duration; they
do not alone establish the accuracy of finite-impedance waveguide modes or
long-duration stability for arbitrary models.

## Internal slab coverage

The existing CPU PML kernels use half-open transverse loops for both E and H.
An internal slab must extend at least one cell into the excluded impedance
volume beyond an upper transverse PMC wall. A slab ending exactly on that
plane omits both tangential E and normal H corrections. The geometry helper
rejects this uncovered-wall configuration with an explanatory error.

## Reproduce

```text
python -m testing.validation.sibc_based_pmc.pml_mirror --steps 200
python -m pytest tests/impedance_surfaces/test_pmc_pml.py -q
```

The regression suite additionally checks normal-to-wall PML rejection,
nonuniform extrusion, uncovered upper transverse faces, unsupported host
loss/dispersion, and acceptance of passive constant and rational surface
loads by the geometry preparation helper.

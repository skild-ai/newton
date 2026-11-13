# Pelican bicycle

Original CAD mechanism created by liinbor (LuckyIYI) of the Skild AI team.
Converted to a self-contained binary USD for this example; no external asset
download, STL loading, or MJCF conversion is needed at runtime. The original
CAD/MJCF source is retained under the permanent
[`steel-pelican-cad-source-v1` tag](https://github.com/skild-ai/newton/tree/steel-pelican-cad-source-v1/newton/examples/assets/steel_tabletop/pelican_bicycle),
independently of this feature branch or a squash merge. It is provenance only;
the example loads the local USD and does not download the source bundle.

The USD embeds the visual meshes and convex collision geometry at the tabletop
scale (2.5 times the original mechanism). Authored masses, inertias, joint
limits, contact filtering, and both closed ankle linkages are preserved.
The front wheel is the only motorized mechanism joint; its speed comes from
the measured position of the separate physical remote fader.

The creator confirmed in skild-ai/newton#27 that the asset is Skild AI's and
that no third-party models, textures, fonts or reference geometry were used.

SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI

SPDX-License-Identifier: Apache-2.0

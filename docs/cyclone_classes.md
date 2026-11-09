# Cyclone 3DR Classification Codes and Custom Model Mapping

Source: Cyclone 3DR class dropdown (screenshots, 2026-10-28).
Names marked with (r) were truncated in the screenshot and reconstructed. Verify them in Cyclone.

## 1. Custom model classes mapped to Cyclone codes

| Model ID | Class | Cyclone code | Cyclone name |
|---|---|---|---|
| 0 | Other | 64 | Other |
| 1 | Ground | 2 | Ground |
| 2 | Paved surface | 71 | Sealed terrain |
| 3 | Low vegetation | 3 | Low Vegetation |
| 4 | Tree | 5 | High Vegetation |
| 5 | Building | 6 | Building |
| 6 | Vehicle | 91 | Vehicle |
| 7 | Construction | 151 | Worksite equipment |
| 8 | Curb | 74 | Curbstone |
| 9 | Road marking | 82 | Custom 82 (rename to "Road marking") |
| 10 | Utility cover | 75 | Manhole |
| 255 | Ignore | not exported | |

Point states in SAMcloud export:

| State | Cyclone code |
|---|---|
| No image saw the point | 0 Created, never classified |
| Model below confidence threshold | 1 Unclassified |

Alternatives for a construction-focused set: 180 HCS construction material, 181 HCS construction vehicle, 182 HCS manhole, 184 HCS curb, 186 HCS material stockpile.

## 2. Full Cyclone class list

| Code | Name |
|---|---|
| 0 | Created, never classified |
| 1 | Unclassified |
| 2 | Ground |
| 3 | Low Vegetation |
| 4 | Medium Vegetation |
| 5 | High Vegetation |
| 6 | Building |
| 7 | Low Point (Noise) |
| 8 | Reserved |
| 9 | Water |
| 10 | Rail |
| 11 | Road Surface |
| 12 | Reserved |
| 13 | Wire - Guard |
| 14 | Wire - Conductor |
| 15 | Transmission Tower |
| 16 | Wire-structure Connector (r) |
| 17 | Bridge Deck |
| 18 | High Noise |
| 19 | Overhead Structure |
| 20 | Ignored Ground |
| 21 | Snow |
| 22 | Temporal Exclusion |
| 23 to 63 | Reserved |
| 64 | Other |
| 65 | Artifact |
| 66 | Hardscape |
| 67 | Moving object |
| 68 | Reflection |
| 69 | Interior points |
| 70 | Natural terrain |
| 71 | Sealed terrain |
| 72 | Retaining wall |
| 73 | Side walk |
| 74 | Curbstone |
| 75 | Manhole |
| 76 | Manmade object |
| 77 | Guard rails |
| 78 | Street furniture |
| 79 | Traffic light |
| 80 | Street lamp |
| 81 | Rail/Road sign |
| 82 | Custom 82 |
| 83 | Custom 83 |
| 84 | Pole |
| 85 | Tunnel |
| 86 | Bridge |
| 87 | Trackbed |
| 88 | Platform |
| 89 | Sleeper |
| 90 | Level crossing |
| 91 | Vehicle |
| 92 | Vegetation |
| 93 | Tree |
| 94 | Shrub |
| 95 | Trunk |
| 96 | Roof |
| 97 | Roof structures |
| 98 | Facades |
| 99 | Balcony |
| 100 | Wall |
| 101 | Ceiling |
| 102 | Floor |
| 103 | Door |
| 104 | Window |
| 105 | Stairs |
| 106 | Ramp |
| 107 | Railing |
| 108 | Column |
| 109 | Beam |
| 110 | Steel structure |
| 111 | Pipe |
| 112 | Wire |
| 113 | Electrical |
| 114 | HVAC |
| 115 | Plumbing |
| 116 | Ceiling structures |
| 117 | Furniture |
| 118 | Chair |
| 119 | Table (Desk) |
| 120 to 125 | Custom 120 to Custom 125 |
| 126 | Stairs wall |
| 127 to 129 | Custom 127 to Custom 129 |
| 130 | Steel structure vertical |
| 131 | Steel structure horizontal (r) |
| 132 | Steel structure diagonal |
| 133 | Steel connection element (r) |
| 134 to 137 | Custom 134 to Custom 137 |
| 138 | Footing |
| 139 | Radiator |
| 140 to 143 | Custom 140 to Custom 143 |
| 144 | Inclined roof |
| 145 | Flat roof |
| 146 to 147 | Custom 146 to Custom 147 |
| 148 | Valve handle |
| 149 | Valve body |
| 150 | Indoor vegetation |
| 151 | Worksite equipment |
| 152 | Noise indoor |
| 153 | Noise outside |
| 154 | Window sill |
| 155 | Window blinds |
| 156 | Inclined ceiling |
| 157 | Outside wall |
| 158 | Custom 158 |
| 159 | Wire installation |
| 160 | Pipe installation |
| 161 | Window noise |
| 162 | Custom 162 |
| 163 | Air duct |
| 164 | Custom 164 |
| 165 | Wall Skeleton |
| 166 to 169 | Custom 166 to Custom 169 |
| 170 | HCS vegetation |
| 171 | HCS manmade object |
| 172 | HCS asphalt road |
| 173 | HCS asphalt hardscape |
| 174 | HCS concrete road |
| 175 | HCS concrete hardscape (r) |
| 176 | HCS street furniture |
| 177 | HCS fencing |
| 178 | HCS trenching |
| 179 | HCS underground utilities (r) |
| 180 | HCS construction material (r) |
| 181 | HCS construction vehicle (r) |
| 182 | HCS manhole |
| 183 | HCS erosion control |
| 184 | HCS curb |
| 185 | HCS foundation |
| 186 | HCS material stockpile |
| 187 to 190 | Custom 187 to Custom 190 |
| 191 | Pipe Bend |
| 192 | Pipe Cylinder |
| 193 | Pipe Connection |
| 194 | Pipe Reducer |
| 195 | Cylinder Tee |
| 196 | Pipe flange |
| 197 | Pipe equipment |
| 198 | Pipe asset |
| 199 | Tank |
| 200 | Pipe instrument |
| 201 | Pipe special |
| 202 to 255 | Custom 202 to Custom 255 |

## 3. SAMcloud export mapping

The former collision in `export_las.py` is fixed. SAM3 profiles now use the effective `classes.json` mapping, including 151 for construction machinery, 72 for outdoor walls and 180 for construction material. Indoor appliances and sanitary fixtures use Custom 120 and Custom 121. Project-defined classes are allocated only from free Cyclone Custom ranges (excluding 82, which is reserved for FlurSeg road markings). Every exported LAS records its model profile and mode.

The indoor set (75 to 82) has the same problem.

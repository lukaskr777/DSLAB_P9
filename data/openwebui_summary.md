# Tables summary

## Contents
- [alembic_version](#alembic_version)  (1x1)
- [channel](#channel)  (0x10)
- [channel_member](#channel_member)  (0x4)
- [chatidtag](#chatidtag)  (0x5)
- [config](#config)  (1x5)
- [document](#document)  (0x8)
- [document_chunk](#document_chunk)  (27927x5)
- [file](#file)  (3584x10)
- [folder](#folder)  (58x10)
- [function](#function)  (5x11)
- [group](#group)  (1x10)
- [knowledge](#knowledge)  (1x9)
- [message_reaction](#message_reaction)  (0x5)
- [migratehistory](#migratehistory)  (18x3)
- [model](#model)  (33x10)
- [prompt](#prompt)  (0x7)
- [tag](#tag)  (120202x4)
- [tool](#tool)  (7x10)
- [user](#user)  (23787x16)

## alembic_version
*Rows*: **1**  •  *Columns*: **1**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| version_num | object | 100.0 | 1 | a5c220713937 |

### Preview
| version_num   |
|:--------------|
| a5c220713937  |

## channel
*Rows*: **0**  •  *Columns*: **10**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## channel_member
*Rows*: **0**  •  *Columns*: **4**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## chatidtag
*Rows*: **0**  •  *Columns*: **5**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## config
*Rows*: **1**  •  *Columns*: **5**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | int32 | 100.0 | 1 | 1 |
| data | object | 100.0 | 1 | {"version": 0, "ui": {"prompt_suggestions": [{"title":… |
| version | int32 | 100.0 | 1 | 0 |
| created_at | object | 100.0 | 1 | 2025-08-30 07:33:30.292887 |
| updated_at | object | 100.0 | 1 | 2025-10-01 02:53:26.629453 |

### Preview
|   id | data                                                                           |   version | created_at                 | updated_at                 |
|-----:|:-------------------------------------------------------------------------------|----------:|:---------------------------|:---------------------------|
|    1 | {"version": 0, "ui": {"prompt_suggestions": [{"title": ["Tell me about public… |         0 | 2025-08-30 07:33:30.292887 | 2025-10-01 02:53:26.629453 |

## document
*Rows*: **0**  •  *Columns*: **8**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## document_chunk
*Rows*: **27927**  •  *Columns*: **5**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 27927 | 9a0bd02e-b2db-4f7f-a891-e9dcb2bd9c8c |
| vector | object | 100.0 | 25760 | [0.02381897,0.044067383,0.001326561,-… |
| collection_name | object | 100.0 | 2288 | file-799269eb-d441-41c8-9adc-2b416b9cb28b |
| text | object | 100.0 | 21566 | /ITA… |
| vmetadata | object | 100.0 | 27758 | {"hash":… |

### Preview
| id                                   | vector                                                                        | collection_name                           | text   | vmetadata                                                                     |
|:-------------------------------------|:------------------------------------------------------------------------------|:------------------------------------------|:-------|:------------------------------------------------------------------------------|
| 9a0bd02e-b2db-4f7f-a891-e9dcb2bd9c8c | [0.02381897,0.044067383,0.001326561,-0.008308411,0.0051994324,0.027130127,-…  | file-799269eb-d441-41c8-9adc-2b416b9cb28b | /ITA…  | {"hash": "5ce7edd05cea7856b7bcd83f5a2f6bdb7afbde134bec61c927afb0d675a363ed",… |
| 4354ac25-8769-4898-b7bf-2da1c9d4a622 | [0.04336548,0.046051025,-0.021316528,0.023544312,0.017349243,0.043304443,-…   | file-799269eb-d441-41c8-9adc-2b416b9cb28b | /JPN…  | {"hash": "5ce7edd05cea7856b7bcd83f5a2f6bdb7afbde134bec61c927afb0d675a363ed",… |
| 3d634f3b-8ec5-4a7c-ae49-9285e9679718 | [0.030014038,0.053619385,-0.024902344,0.021896362,0.0090789795,0.013000488,-… | file-799269eb-d441-41c8-9adc-2b416b9cb28b | /LTH…  | {"hash": "5ce7edd05cea7856b7bcd83f5a2f6bdb7afbde134bec61c927afb0d675a363ed",… |
| b3cbccb9-3b57-4587-9c07-8f414da2923f | [0.012962341,0.042541504,-0.020507812,0.015106201,0.003967285,-0.005405426,-… | file-799269eb-d441-41c8-9adc-2b416b9cb28b | /LVI…  | {"hash": "5ce7edd05cea7856b7bcd83f5a2f6bdb7afbde134bec61c927afb0d675a363ed",… |
| d49e2219-0872-40b9-adcc-c56b35f1a453 | [0.038879395,0.053375244,-0.013435364,0.028381348,0.013763428,0.03540039,-…   | file-799269eb-d441-41c8-9adc-2b416b9cb28b | /NOR…  | {"hash": "5ce7edd05cea7856b7bcd83f5a2f6bdb7afbde134bec61c927afb0d675a363ed",… |

## file
*Rows*: **3584**  •  *Columns*: **10**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 3584 | bb93d52f-6901-4f70-9163-1fc85cca6a84 |
| user_id | object | 100.0 | 1399 | 97d92cc8-0591-4fa0-afdb-969833de56eb |
| filename | object | 100.0 | 2691 | CustomerInfo.pdf |
| meta | object | 100.0 | 3186 | {"name": "CustomerInfo.pdf", "content_type":… |
| created_at | int64 | 100.0 | 3096 | 1757409875 |
| hash | object | 98.2 | 2615 | … |
| data | object | 100.0 | 2676 | {"status": "completed", "content": "OBI Schweiz… |
| updated_at | int64 | 100.0 | 3096 | 1757409875 |
| path | object | 100.0 | 3584 | s3://publicai-bucket/uploads/bb93d52f-6901-4f70-9163-… |
| access_control | object | 100.0 | 1 | null |

### Preview
| id                                   | user_id                              | filename                                          | meta                                                                             |   created_at | hash                                                             | data                                                                          |   updated_at | path                                                                         | access_control   |
|:-------------------------------------|:-------------------------------------|:--------------------------------------------------|:---------------------------------------------------------------------------------|-------------:|:-----------------------------------------------------------------|:------------------------------------------------------------------------------|-------------:|:-----------------------------------------------------------------------------|:-----------------|
| bb93d52f-6901-4f70-9163-1fc85cca6a84 | 97d92cc8-0591-4fa0-afdb-969833de56eb | CustomerInfo.pdf                                  | {"name": "CustomerInfo.pdf", "content_type": "application/pdf", "size": 397128,… |   1757409875 | de91a83fd0e6fedfccf593f2c03377716a4c13d8afe6d48de317f289e5430b2e | {"status": "completed", "content": "OBI Schweiz GmbH\nRheinweg 11\n8200…      |   1757409875 | s3://publicai-…                                                              | null             |
| d52bc36f-d598-4b05-8df9-22c614a7c037 | 26edea9f-ffa5-465b-ad88-bc824f61779d | TEXTOS.txt                                        | {"name": "TEXTOS.txt", "content_type": "text/plain", "size": 13840, "data": {},… |   1757433498 | ee94d1a3f23515c983b08a03ea038de61cf4ec2a0ac00a3c709ec81accd7d488 | {"status": "completed", "content": "\u00bfEsp\u00eda Huawei para el gobierno… |   1757433498 | s3://publicai-bucket/uploads/d52bc36f-d598-4b05-8df9-22c614a7c037_TEXTOS.txt | null             |
| 9b333096-d27a-4b24-a032-73a2e0c22678 | 81cb5573-27c3-4d2a-813f-8b84cf838659 | 250822_Präsentation Anlass 2 Standards_EMBAG.pptx | {"name": "250822_Pr\u00e4sentation Anlass 2 Standards_EMBAG.pptx",…              |   1757515640 | 747120e434389c577c0ee91c76bc0a0114f9ae7f5e16320d348d49846905a14f | {"status": "completed", "content": "PowerPoint-Pr\u00e4sentation\n\nEMBAG…    |   1757515640 | s3://publicai-…                                                              | null             |
| 97f81a73-575c-45a8-8ae0-af997c81ce9a | c3bf5865-39a5-4449-8418-4d65930613ba | Pasted_Text_1757454840559.txt                     | {"name": "Pasted_Text_1757454840559.txt", "content_type": "text/plain", "size":… |   1757454842 | 9375c71fdaaff88fc50670e9b37dba3749e27a5e76486922a8a625b1860da692 | {"status": "completed", "content": "[9/9 18:09] Nadson Por\u00e3o Natural:…   |   1757454842 | s3://publicai-bucket/uploads/97f81a73-575c-45a8-8ae0-…                       | null             |
| 6dbd2ab6-5e60-49a8-9ff8-975e2a78503b | 32f30fe6-d288-45ae-be25-aebcb841dbb7 | CV.pdf                                            | {"name": "CV.pdf", "content_type": "application/pdf", "size": 359039, "data":…   |   1757510322 | 55873c3f7dbf5b2fdfac35c4dbe44165a6a7b0051ee6ea6ef6aad9e58a23f498 | {"status": "completed", "content": "CV Fran\u00e7ais Simple Moderne…          |   1757510322 | s3://publicai-bucket/uploads/6dbd2ab6-5e60-49a8-9ff8-975e2a78503b_CV.pdf     | null             |

## folder
*Rows*: **58**  •  *Columns*: **10**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 58 | e924635d-2ccb-4cf7-b276-dddfdaaa2bbc |
| parent_id | object | 0.0 | 0 | — |
| user_id | object | 100.0 | 37 | 37e1e3a4-5a6c-41d8-b59f-e4f66ead2b46 |
| name | object | 100.0 | 57 | Progress & Change candidacies |
| items | object | 100.0 | 1 | null |
| meta | object | 100.0 | 2 | null |
| is_expanded | bool | 100.0 | 2 | False |
| created_at | int64 | 100.0 | 58 | 1756891621 |
| updated_at | int64 | 100.0 | 47 | 1757573030 |
| data | object | 100.0 | 15 | {"system_prompt": "", "files": []} |

### Preview
| id                                   | parent_id   | user_id                              | name                                     | items   | meta   | is_expanded   |   created_at |   updated_at | data                                                             |
|:-------------------------------------|:------------|:-------------------------------------|:-----------------------------------------|:--------|:-------|:--------------|-------------:|-------------:|:-----------------------------------------------------------------|
| e924635d-2ccb-4cf7-b276-dddfdaaa2bbc |             | 37e1e3a4-5a6c-41d8-b59f-e4f66ead2b46 | Progress & Change candidacies            | null    | null   | False         |   1756891621 |   1757573030 | {"system_prompt": "", "files": []}                               |
| 89c67288-568e-455b-bd0f-381ba4f12045 |             | 37e1e3a4-5a6c-41d8-b59f-e4f66ead2b46 | ECNL                                     | null    | null   | False         |   1756890170 |   1757573030 | {"system_prompt": "", "files": []}                               |
| 361a4ea8-2d3e-4aeb-84fd-60c36089a5f2 |             | 2d1e2431-b75d-41e1-8aa6-97309451070c | AAIA                                     | null    | null   | False         |   1759047308 |   1759051995 | {"system_prompt": "", "files": [{"type": "file", "file": {"id":… |
| aa566294-abb1-461a-a6e8-1d2e9fbedb62 |             | 37e1e3a4-5a6c-41d8-b59f-e4f66ead2b46 | NGO Boards                               | null    | null   | False         |   1756886202 |   1757573030 | {"system_prompt": "", "files": []}                               |
| 541b2645-4068-49df-af11-b6f5238565b8 |             | 37e1e3a4-5a6c-41d8-b59f-e4f66ead2b46 | Campaign to End Repression in Azerbaijan | null    | null   | False         |   1756886180 |   1757573030 | {"system_prompt": "", "files": []}                               |

## function
*Rows*: **5**  •  *Columns*: **11**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 5 | singlish_toggle |
| user_id | object | 100.0 | 1 | bc49bba7-3343-4f03-8601-a4c278228c24 |
| name | object | 100.0 | 5 | Singlish Toggle |
| type | object | 100.0 | 1 | filter |
| content | object | 100.0 | 5 | from pydantic import BaseModel, Field\nfrom typing import… |
| meta | object | 100.0 | 5 | {"description": " Enables Singlish responses with authentic… |
| created_at | int64 | 100.0 | 5 | 1756683098 |
| updated_at | int64 | 100.0 | 5 | 1756683347 |
| valves | object | 100.0 | 1 | null |
| is_active | bool | 100.0 | 2 | True |
| is_global | bool | 100.0 | 2 | False |

### Preview
| id                         | user_id                              | name                       | type   | content                                                                        | meta                                                                          |   created_at |   updated_at | valves   | is_active   | is_global   |
|:---------------------------|:-------------------------------------|:---------------------------|:-------|:-------------------------------------------------------------------------------|:------------------------------------------------------------------------------|-------------:|-------------:|:---------|:------------|:------------|
| singlish_toggle            | bc49bba7-3343-4f03-8601-a4c278228c24 | Singlish Toggle            | filter | from pydantic import BaseModel, Field\nfrom typing import Optional\nimport…    | {"description": " Enables Singlish responses with authentic Singaporean…      |   1756683098 |   1756683347 | null     | True        | False       |
| debug_filter               | bc49bba7-3343-4f03-8601-a4c278228c24 | Debug Filter               | filter | import json\nfrom typing import Optional\n\n\nclass Filter:\n def inlet(self,… | {"description": "Dumps raw response", "manifest": {}}                         |   1756683855 |   1757518730 | null     | True        | False       |
| schwizerdütsch_toggle      | bc49bba7-3343-4f03-8601-a4c278228c24 | Schwiizerdütsch Toggle     | filter | from pydantic import BaseModel, Field\nfrom typing import Optional\nimport…    | {"description": "Enables Swiss German responses with authentic dialect…       |   1756683428 |   1756788131 | null     | True        | False       |
| sponsor_attribution_filter | bc49bba7-3343-4f03-8601-a4c278228c24 | Sponsor Attribution Filter | filter | from typing import Optional\nimport random\n\n\nclass Filter:\n def…           | {"description": "Attributes our compute sponsors", "manifest": {}}            |   1756685749 |   1758538500 | null     | True        | True        |
| debug_filter_v2            | bc49bba7-3343-4f03-8601-a4c278228c24 | Debug Filter V2            | filter | """\ntitle: debug_filter\nauthor: thiswillbeyourightub\nauthor_url:…           | {"description": "Debugging", "manifest": {"title": "debug_filter", "author":… |   1757518628 |   1758905955 | null     | False       | False       |

## group
*Rows*: **1**  •  *Columns*: **10**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 1 | 88fb4519-541f-48ca-b8f5-b5f3c58c00e8 |
| user_id | object | 100.0 | 1 | bc49bba7-3343-4f03-8601-a4c278228c24 |
| name | object | 100.0 | 1 | Plus |
| description | object | 100.0 | 1 | The Public AI Plus Experience |
| data | object | 100.0 | 1 | null |
| meta | object | 100.0 | 1 | null |
| permissions | object | 100.0 | 1 | {"workspace": {"models": true, "knowledge": true,… |
| user_ids | object | 100.0 | 1 | ["574bcca6-4e1a-46b1-ac4c-731398842e21",… |
| created_at | int64 | 100.0 | 1 | 1757942843 |
| updated_at | int64 | 100.0 | 1 | 1759007241 |

### Preview
| id                                   | user_id                              | name   | description                   | data   | meta   | permissions                                                                  | user_ids                                  |   created_at |   updated_at |
|:-------------------------------------|:-------------------------------------|:-------|:------------------------------|:-------|:-------|:-----------------------------------------------------------------------------|:------------------------------------------|-------------:|-------------:|
| 88fb4519-541f-48ca-b8f5-b5f3c58c00e8 | bc49bba7-3343-4f03-8601-a4c278228c24 | Plus   | The Public AI Plus Experience | null   | null   | {"workspace": {"models": true, "knowledge": true, "prompts": true, "tools":… | ["574bcca6-4e1a-46b1-ac4c-731398842e21",… |   1757942843 |   1759007241 |

## knowledge
*Rows*: **1**  •  *Columns*: **9**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 1 | 2b566631-f8e1-4c25-b7dd-8b3bcb116107 |
| user_id | object | 100.0 | 1 | bc49bba7-3343-4f03-8601-a4c278228c24 |
| name | object | 100.0 | 1 | Test Knowledge |
| description | object | 100.0 | 1 | Testing |
| data | object | 100.0 | 1 | {"file_ids": ["0aad2bf8-a1f1-44c1-a24d-0668eb37eb05",… |
| meta | object | 100.0 | 1 | null |
| created_at | int64 | 100.0 | 1 | 1756562078 |
| updated_at | int64 | 100.0 | 1 | 1759287401 |
| access_control | object | 100.0 | 1 | {"read": {"group_ids": [], "user_ids": []}, "write":… |

### Preview
| id                                   | user_id                              | name           | description   | data                                                   | meta   |   created_at |   updated_at | access_control                                                          |
|:-------------------------------------|:-------------------------------------|:---------------|:--------------|:-------------------------------------------------------|:-------|-------------:|-------------:|:------------------------------------------------------------------------|
| 2b566631-f8e1-4c25-b7dd-8b3bcb116107 | bc49bba7-3343-4f03-8601-a4c278228c24 | Test Knowledge | Testing       | {"file_ids": ["0aad2bf8-a1f1-44c1-a24d-0668eb37eb05",… | null   |   1756562078 |   1759287401 | {"read": {"group_ids": [], "user_ids": []}, "write": {"group_ids": [],… |

## message_reaction
*Rows*: **0**  •  *Columns*: **5**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## migratehistory
*Rows*: **18**  •  *Columns*: **3**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | int32 | 100.0 | 18 | 1 |
| name | object | 100.0 | 18 | 001_initial_schema |
| migrated_at | object | 100.0 | 18 | 2025-08-30 03:57:21.818791 |

### Preview
|   id | name                  | migrated_at                |
|-----:|:----------------------|:---------------------------|
|    1 | 001_initial_schema    | 2025-08-30 03:57:21.818791 |
|    2 | 002_add_local_sharing | 2025-08-30 03:57:21.829027 |
|    3 | 003_add_auth_api_key  | 2025-08-30 03:57:21.835038 |
|    4 | 004_add_archived      | 2025-08-30 03:57:21.838858 |
|    5 | 005_add_updated_at    | 2025-08-30 03:57:21.852714 |

## model
*Rows*: **33**  •  *Columns*: **10**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 33 | aisingapore/Gemma-SEA-LION-v4-27B-IT-quantized |
| user_id | object | 100.0 | 3 | bc49bba7-3343-4f03-8601-a4c278228c24 |
| base_model_id | object | 3.0 | 1 | swiss-ai/apertus-70b-instruct |
| name | object | 100.0 | 26 | aisingapore/Gemma-SEA-LION-v4-27B-IT-quantized |
| meta | object | 100.0 | 11 | {"profile_image_url": "/static/favicon.png", "description":… |
| params | object | 100.0 | 11 | {} |
| created_at | int64 | 100.0 | 32 | 1756526296 |
| updated_at | int64 | 100.0 | 32 | 1756526296 |
| access_control | object | 100.0 | 4 | {} |
| is_active | bool | 100.0 | 2 | False |

### Preview
| id                                             | user_id                              | base_model_id   | name                                           | meta                                                               | params   |   created_at |   updated_at | access_control   | is_active   |
|:-----------------------------------------------|:-------------------------------------|:----------------|:-----------------------------------------------|:-------------------------------------------------------------------|:---------|-------------:|-------------:|:-----------------|:------------|
| aisingapore/Gemma-SEA-LION-v4-27B-IT-quantized | bc49bba7-3343-4f03-8601-a4c278228c24 |                 | aisingapore/Gemma-SEA-LION-v4-27B-IT-quantized | {"profile_image_url": "/static/favicon.png", "description": null,… | {}       |   1756526296 |   1756526296 | {}               | False       |
| cohere-embed-multilingual-v3                   | bc49bba7-3343-4f03-8601-a4c278228c24 |                 | cohere-embed-multilingual-v3                   | {"profile_image_url": "/static/favicon.png", "description": null,… | {}       |   1756526297 |   1756526297 | {}               | False       |
| cohere-rerank-v3-5                             | bc49bba7-3343-4f03-8601-a4c278228c24 |                 | cohere-rerank-v3-5                             | {"profile_image_url": "/static/favicon.png", "description": null,… | {}       |   1756526298 |   1756526298 | {}               | False       |
| tinyllama-instruct                             | bc49bba7-3343-4f03-8601-a4c278228c24 |                 | tinyllama-instruct                             | {"profile_image_url": "/static/favicon.png", "description": null,… | {}       |   1756526302 |   1756526302 | {}               | False       |
| llama3-2-3b-instruct                           | bc49bba7-3343-4f03-8601-a4c278228c24 |                 | Task Model                                     | {"profile_image_url": "/static/favicon.png", "description": null,… | {}       |   1756526309 |   1756526309 | null             | True        |

## prompt
*Rows*: **0**  •  *Columns*: **7**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## tag
*Rows*: **120202**  •  *Columns*: **4**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 25747 | technology |
| name | object | 100.0 | 25995 | Technology |
| user_id | object | 100.0 | 19694 | bc49bba7-3343-4f03-8601-a4c278228c24 |
| meta | object | 100.0 | 1 | null |

### Preview
| id                      | name                    | user_id                              | meta   |
|:------------------------|:------------------------|:-------------------------------------|:-------|
| technology              | Technology              | bc49bba7-3343-4f03-8601-a4c278228c24 | null   |
| business                | Business                | bc49bba7-3343-4f03-8601-a4c278228c24 | null   |
| singapore               | Singapore               | bc49bba7-3343-4f03-8601-a4c278228c24 | null   |
| general                 | General                 | bc49bba7-3343-4f03-8601-a4c278228c24 | null   |
| artificial_intelligence | Artificial Intelligence | bc49bba7-3343-4f03-8601-a4c278228c24 | null   |

## tool
*Rows*: **7**  •  *Columns*: **10**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 7 | test_smo |
| user_id | object | 100.0 | 6 | 375cf9f4-6ae2-4e3a-86b5-820b86f97f43 |
| name | object | 100.0 | 7 | Test SMO |
| content | object | 100.0 | 4 | import os\nimport requests\nfrom datetime import… |
| specs | object | 100.0 | 3 | [{"name": "calculator", "description": "Calculate the… |
| meta | object | 100.0 | 7 | {"description": "Premier outil de test", "manifest": {}} |
| created_at | int64 | 100.0 | 7 | 1758784656 |
| updated_at | int64 | 100.0 | 7 | 1758905970 |
| valves | object | 100.0 | 1 | null |
| access_control | object | 100.0 | 2 | {} |

### Preview
| id               | user_id                              | name             | content                                                                          | specs                                                                           | meta                                                                        |   created_at |   updated_at | valves   | access_control                                                          |
|:-----------------|:-------------------------------------|:-----------------|:---------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|:----------------------------------------------------------------------------|-------------:|-------------:|:---------|:------------------------------------------------------------------------|
| test_smo         | 375cf9f4-6ae2-4e3a-86b5-820b86f97f43 | Test SMO         | import os\nimport requests\nfrom datetime import datetime\nfrom pydantic import… | [{"name": "calculator", "description": "Calculate the result of an equation.",… | {"description": "Premier outil de test", "manifest": {}}                    |   1758784656 |   1758905970 | null     | {}                                                                      |
| generator_imagen | 64808981-8e82-4343-8f5c-c0ad3205286f | Generator imagen | import os\nimport requests\nfrom datetime import datetime\nfrom pydantic import… | [{"name": "calculator", "description": "Calculate the result of an equation.",… | {"description": "Gerador de iamgem", "manifest": {}}                        |   1759028036 |   1759028036 | null     | {"read": {"group_ids": [], "user_ids": []}, "write": {"group_ids": [],… |
| my_tool          | 119ebd55-b277-4d67-831a-c08b33bf0df6 | my tool          | import os\nimport requests\nfrom datetime import datetime\nfrom pydantic import… | [{"name": "calculator", "description": "Calculate the result of an equation.",… | {"description": "my tool", "manifest": {}}                                  |   1758388227 |   1758388265 | null     | {"read": {"group_ids": [], "user_ids": []}, "write": {"group_ids": [],… |
| ds               | d7d217f2-8894-4581-8f26-f095a5e874bd | Weather copied   | """\ntitle: Keyless Weather\nauthor: spyci\nauthor_url:…                         | [{"name": "get_current_weather", "description": "Get the current weather for a… | {"description": "asdf", "manifest": {"title": "Keyless Weather", "author":… |   1758276141 |   1758721057 | null     | {}                                                                      |
| test             | f7f2534a-011b-4351-891b-199136ac9a41 | test             | import os\nimport requests\nfrom datetime import datetime\nfrom pydantic import… | [{"name": "calculator", "description": "Calculate the result of an equation.",… | {"description": "test", "manifest": {}}                                     |   1758718880 |   1759168357 | null     | {}                                                                      |

## user
*Rows*: **23787**  •  *Columns*: **16**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 23787 | fa8cb001-8c32-405a-a257-e6860535d510 |
| name | object | 100.0 | 23563 | name_8205da8f |
| email | object | 100.0 | 23787 | email_925cd6bb |
| role | object | 100.0 | 2 | user |
| profile_image_url | object | 0.0 | 0 | — |
| api_key | object | 0.0 | 0 | — |
| created_at | int64 | 100.0 | 23233 | 1757256831 |
| updated_at | int64 | 100.0 | 23233 | 1757256831 |
| last_active_at | int64 | 100.0 | 23493 | 1758136576 |
| settings | object | 0.0 | 0 | — |
| info | object | 0.0 | 0 | — |
| oauth_sub | object | 0.0 | 0 | — |
| username | object | 0.0 | 0 | — |
| bio | object | 0.0 | 0 | — |
| gender | object | 0.0 | 0 | — |
| date_of_birth | object | 0.0 | 0 | — |

### Preview
| id                                   | name          | email          | role   | profile_image_url   | api_key   |   created_at |   updated_at |   last_active_at | settings   | info   | oauth_sub   | username   | bio   | gender   | date_of_birth   |
|:-------------------------------------|:--------------|:---------------|:-------|:--------------------|:----------|-------------:|-------------:|-----------------:|:-----------|:-------|:------------|:-----------|:------|:---------|:----------------|
| fa8cb001-8c32-405a-a257-e6860535d510 | name_8205da8f | email_925cd6bb | user   |                     |           |   1757256831 |   1757256831 |       1758136576 |            |        |             |            |       |          |                 |
| 9604a1bd-3291-4854-869d-2e0ea9e37711 | name_99b36a42 | email_397d577e | user   |                     |           |   1757668142 |   1757668142 |       1757862574 |            |        |             |            |       |          |                 |
| 07313d86-b43b-478f-acea-d6a397ed89bd | name_098fbeac | email_098fbeac | user   |                     |           |   1757922953 |   1757922953 |       1757940825 |            |        |             |            |       |          |                 |
| bf08b4c8-80aa-49e3-ba6d-654f9a06c1b5 | name_8101e0e7 | email_02039673 | user   |                     |           |   1757522258 |   1757522258 |       1757695557 |            |        |             |            |       |          |                 |
| f7d6f7b5-3bab-463d-a527-27cf90687c53 | name_687eca73 | email_687eca73 | user   |                     |           |   1757548198 |   1757548198 |       1757716575 |            |        |             |            |       |          |                 |


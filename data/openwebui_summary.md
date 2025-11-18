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
| id | object | 100.0 | 27927 | 079181ca-5f66-492f-856b-63f682709678 |
| vector | object | 100.0 | 25760 | [-0.008598328,-0.0019521713,-0.056274414,-… |
| collection_name | object | 100.0 | 2288 | file-0aad2bf8-a1f1-44c1-a24d-0668eb37eb05 |
| text | object | 100.0 | 21566 | State of DAOs in Malaysia Shared Version 1\n\n\nState of… |
| vmetadata | object | 100.0 | 27758 | {"hash":… |

### Preview
| id                                   | vector                                                                       | collection_name                           | text                                                                             | vmetadata                                                                     |
|:-------------------------------------|:-----------------------------------------------------------------------------|:------------------------------------------|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------|
| 079181ca-5f66-492f-856b-63f682709678 | [-0.008598328,-0.0019521713,-0.056274414,-…                                  | file-0aad2bf8-a1f1-44c1-a24d-0668eb37eb05 | State of DAOs in Malaysia Shared Version 1\n\n\nState of DAOs in Malaysia…       | {"hash": "120961eda5f8db54126338a96f465f2485b6a098aad285a3d981a888942afa10",… |
| 101e228a-9a5e-4c78-866d-39c196f5c885 | [-0.016693115,-0.0042686462,-0.03387451,-…                                   | file-0aad2bf8-a1f1-44c1-a24d-0668eb37eb05 | This report provides the following: 1) A working definition of DAOs adopted…     | {"hash": "120961eda5f8db54126338a96f465f2485b6a098aad285a3d981a888942afa10",… |
| f3aa4e39-605e-44be-86ad-fa973d099e0a | [0.009346008,0.011619568,0.014259338,0.008079529,-0.005722046,0.011833191,-… | file-0aad2bf8-a1f1-44c1-a24d-0668eb37eb05 | -Biases and limitations. \n\nDefining DAOs \n\nThe Malaysian Context \nNational… | {"hash": "120961eda5f8db54126338a96f465f2485b6a098aad285a3d981a888942afa10",… |
| a34f7921-5313-445f-b251-a1acd2e69321 | [0.018936157,0.014678955,0.0062675476,0.05307007,-0.03265381,-0.005302429,-… | file-0aad2bf8-a1f1-44c1-a24d-0668eb37eb05 | Two policy documents illustrate this ambivalent approach: \n\nNational…          | {"hash": "120961eda5f8db54126338a96f465f2485b6a098aad285a3d981a888942afa10",… |
| c934deeb-587c-40fe-bd6f-479c9e12280c | [0.010093689,0.016448975,0.05444336,0.05807495,-0.040924072,0.017242432,-…   | file-0aad2bf8-a1f1-44c1-a24d-0668eb37eb05 | ASEAN Digital Economy Framework Agreement (DEFA) Report (2024): \n\nAs the…      | {"hash": "120961eda5f8db54126338a96f465f2485b6a098aad285a3d981a888942afa10",… |

## file
*Rows*: **3584**  •  *Columns*: **10**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 3584 | 595630ca-3162-41d2-a61e-924bd1fb27a6 |
| user_id | object | 100.0 | 1399 | bc49bba7-3343-4f03-8601-a4c278228c24 |
| filename | object | 100.0 | 2691 | daos_in_malaysia.pdf |
| meta | object | 100.0 | 3186 | {"name": "daos_in_malaysia.pdf", "content_type":… |
| created_at | int64 | 100.0 | 3096 | 1756711111 |
| hash | object | 98.2 | 2615 | … |
| data | object | 100.0 | 2676 | {"status": "failed", "content": "State of DAOs in Malaysia… |
| updated_at | int64 | 100.0 | 3096 | 1756711111 |
| path | object | 100.0 | 3584 | s3://publicai-bucket/uploads/595630ca-3162-41d2-a61e-… |
| access_control | object | 100.0 | 1 | null |

### Preview
| id                                   | user_id                              | filename             | meta                                                                         |   created_at | hash                                                             | data                                                                          |   updated_at | path            | access_control   |
|:-------------------------------------|:-------------------------------------|:---------------------|:-----------------------------------------------------------------------------|-------------:|:-----------------------------------------------------------------|:------------------------------------------------------------------------------|-------------:|:----------------|:-----------------|
| 595630ca-3162-41d2-a61e-924bd1fb27a6 | bc49bba7-3343-4f03-8601-a4c278228c24 | daos_in_malaysia.pdf | {"name": "daos_in_malaysia.pdf", "content_type": "application/pdf", "size":… |   1756711111 | 120961eda5f8db54126338a96f465f2485b6a098aad285a3d981a888942afa10 | {"status": "failed", "content": "State of DAOs in Malaysia Shared Version…    |   1756711111 | s3://publicai-… | null             |
| 7002538a-032b-4251-a375-4d5efbefae09 | fde7ac24-4efc-4874-9e60-fa80fc8b33dc | Exercise2_Task.pdf   | {"name": "Exercise2_Task.pdf", "content_type": "application/pdf", "size":…   |   1756835446 | 8b06063757d7eb6ad151389305da561074726393a56f0d66f1d14515906a0e88 | {"status": "completed", "content": "TKP4140 Process Control\nDepartment of…   |   1756835446 | s3://publicai-… | null             |
| affef471-ce26-486f-afbe-2932d5e91e14 | bc49bba7-3343-4f03-8601-a4c278228c24 | daos_in_malaysia.pdf | {"name": "daos_in_malaysia.pdf", "content_type": "application/pdf", "size":… |   1756710236 | 120961eda5f8db54126338a96f465f2485b6a098aad285a3d981a888942afa10 | {"status": "failed", "content": "State of DAOs in Malaysia Shared Version…    |   1756710236 | s3://publicai-… | null             |
| e344cf20-aebf-4d89-8b9f-65266e22f1eb | bc49bba7-3343-4f03-8601-a4c278228c24 | daos_in_malaysia.pdf | {"name": "daos_in_malaysia.pdf", "content_type": "application/pdf", "size":… |   1756710151 | 120961eda5f8db54126338a96f465f2485b6a098aad285a3d981a888942afa10 | {"status": "completed", "content": "State of DAOs in Malaysia Shared Version… |   1756710151 | s3://publicai-… | null             |
| 99326e9c-31bc-406c-84fb-337d98f4e819 | bc49bba7-3343-4f03-8601-a4c278228c24 | daos_in_malaysia.pdf | {"name": "daos_in_malaysia.pdf", "content_type": "application/pdf", "size":… |   1756709853 | 120961eda5f8db54126338a96f465f2485b6a098aad285a3d981a888942afa10 | {"status": "failed", "content": "State of DAOs in Malaysia Shared Version…    |   1756709853 | s3://publicai-… | null             |

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
| id | object | 100.0 | 23787 | f36f7210-61da-443b-9028-acb1660428bb |
| name | object | 100.0 | 23563 | name_8bd8bf70 |
| email | object | 100.0 | 23787 | email_ec3bdc04 |
| role | object | 100.0 | 2 | user |
| profile_image_url | object | 0.0 | 0 | — |
| api_key | object | 0.0 | 0 | — |
| created_at | int64 | 100.0 | 23233 | 1756833300 |
| updated_at | int64 | 100.0 | 23233 | 1756833300 |
| last_active_at | int64 | 100.0 | 23493 | 1756838966 |
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
| f36f7210-61da-443b-9028-acb1660428bb | name_8bd8bf70 | email_ec3bdc04 | user   |                     |           |   1756833300 |   1756833300 |       1756838966 |            |        |             |            |       |          |                 |
| 3e04b809-385b-462e-9697-0928b5d40951 | name_d8c086b0 | email_d8c086b0 | user   |                     |           |   1756617068 |   1756617068 |       1756617715 |            |        |             |            |       |          |                 |
| 8ca890bc-6b43-4227-b81a-2a0d2b825519 | name_929d2a39 | email_929d2a39 | user   |                     |           |   1756615132 |   1756615132 |       1756616855 |            |        |             |            |       |          |                 |
| 05756b43-a9fe-4c6b-8186-748fe1ab1aca | name_80aba695 | email_80aba695 | user   |                     |           |   1756617815 |   1756617815 |       1756619145 |            |        |             |            |       |          |                 |
| 2ad306f4-7474-492a-96ec-e295bde682e0 | name_60203a71 | email_b32c924a | user   |                     |           |   1756839586 |   1756839586 |       1756839724 |            |        |             |            |       |          |                 |


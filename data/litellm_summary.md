# Tables summary

## Contents
- [LiteLLM_AuditLog](#litellm_auditlog)  (0x9)
- [LiteLLM_BudgetTable](#litellm_budgettable)  (6x13)
- [LiteLLM_Config](#litellm_config)  (1x2)
- [LiteLLM_CronJob](#litellm_cronjob)  (0x5)
- [LiteLLM_DailyTagSpend](#litellm_dailytagspend)  (2945x18)
- [LiteLLM_DailyTeamSpend](#litellm_dailyteamspend)  (717x18)
- [LiteLLM_DailyUserSpend](#litellm_dailyuserspend)  (704x18)
- [LiteLLM_EndUserTable](#litellm_endusertable)  (23805x7)
- [LiteLLM_GuardrailsTable](#litellm_guardrailstable)  (0x6)
- [LiteLLM_HealthCheckTable](#litellm_healthchecktable)  (0x13)
- [LiteLLM_MCPServerTable](#litellm_mcpservertable)  (0x19)
- [LiteLLM_ManagedFileTable](#litellm_managedfiletable)  (0x9)
- [LiteLLM_ManagedObjectTable](#litellm_managedobjecttable)  (0x10)
- [LiteLLM_ManagedVectorStoresTable](#litellm_managedvectorstorestable)  (0x9)
- [LiteLLM_ModelTable](#litellm_modeltable)  (0x6)
- [LiteLLM_ObjectPermissionTable](#litellm_objectpermissiontable)  (1x4)
- [LiteLLM_OrganizationMembership](#litellm_organizationmembership)  (0x7)
- [LiteLLM_OrganizationTable](#litellm_organizationtable)  (0x12)
- [LiteLLM_PromptTable](#litellm_prompttable)  (0x6)
- [LiteLLM_ProxyModelTable](#litellm_proxymodeltable)  (0x8)
- [LiteLLM_SpendLogs](#litellm_spendlogs)  (713244x29)
- [LiteLLM_TeamMembership](#litellm_teammembership)  (0x4)
- [LiteLLM_TeamTable](#litellm_teamtable)  (1x23)
- [LiteLLM_UserNotifications](#litellm_usernotifications)  (0x5)
- [LiteLLM_UserTable](#litellm_usertable)  (4x24)
- [LiteLLM_VerificationToken](#litellm_verificationtoken)  (65x31)
- [_prisma_migrations](#_prisma_migrations)  (35x8)

## LiteLLM_AuditLog
*Rows*: **0**  •  *Columns*: **9**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_BudgetTable
*Rows*: **6**  •  *Columns*: **13**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| budget_id | object | 100.0 | 6 | Public AI Free Tier |
| max_budget | float64 | 16.7 | 1 | 1.0 |
| soft_budget | float64 | 0.0 | 0 | — |
| max_parallel_requests | float64 | 33.3 | 1 | 5.0 |
| tpm_limit | float64 | 16.7 | 1 | 25000.0 |
| rpm_limit | float64 | 50.0 | 2 | 5.0 |
| model_max_budget | object | 0.0 | 0 | — |
| budget_duration | object | 16.7 | 1 | 24h |
| budget_reset_at | object | 33.3 | 2 | 2025-10-01 06:25:55.976 |
| created_at | object | 100.0 | 6 | 2025-08-31 03:11:43.065 |
| created_by | object | 100.0 | 1 | default_user_id |
| updated_at | object | 100.0 | 6 | 2025-08-31 03:11:43.065 |
| updated_by | object | 100.0 | 1 | default_user_id |

### Preview
| budget_id                            |   max_budget |   soft_budget |   max_parallel_requests |   tpm_limit |   rpm_limit | model_max_budget   | budget_duration   | budget_reset_at         | created_at              | created_by      | updated_at              | updated_by      |
|:-------------------------------------|-------------:|--------------:|------------------------:|------------:|------------:|:-------------------|:------------------|:------------------------|:------------------------|:----------------|:------------------------|:----------------|
| Public AI Free Tier                  |          nan |           nan |                     nan |         nan |           5 |                    |                   |                         | 2025-08-31 03:11:43.065 | default_user_id | 2025-08-31 03:11:43.065 | default_user_id |
| bc49bba7-3343-4f03-8601-a4c278228c24 |          nan |           nan |                     nan |         nan |           5 |                    |                   |                         | 2025-08-31 03:49:46.845 | default_user_id | 2025-08-31 04:26:44.141 | default_user_id |
| c1cfc6d4-34c4-46ac-802e-8825a8f96869 |          nan |           nan |                       5 |         nan |         nan |                    |                   |                         | 2025-08-31 04:51:09.495 | default_user_id | 2025-08-31 04:51:09.495 | default_user_id |
| 23f17d35-a4c7-46e3-a843-c8f12350cb68 |          nan |           nan |                       5 |         nan |         nan |                    |                   |                         | 2025-08-31 04:51:41.239 | default_user_id | 2025-08-31 04:51:41.239 | default_user_id |
| public_ai_free                       |            1 |           nan |                     nan |       25000 |          30 |                    | 24h               | 2025-10-01 06:25:55.976 | 2025-08-31 05:04:18.394 | default_user_id | 2025-09-06 11:13:49.038 | default_user_id |

## LiteLLM_Config
*Rows*: **1**  •  *Columns*: **2**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| param_name | object | 100.0 | 1 | model_cost_map_reload_config |
| param_value | object | 100.0 | 1 | {"force_reload": true, "interval_hours": null} |

### Preview
| param_name                   | param_value                                    |
|:-----------------------------|:-----------------------------------------------|
| model_cost_map_reload_config | {"force_reload": true, "interval_hours": null} |

## LiteLLM_CronJob
*Rows*: **0**  •  *Columns*: **5**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_DailyTagSpend
*Rows*: **2945**  •  *Columns*: **18**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 2945 | cce6c2d8-5f87-4e11-bf5d-c9e0c5518ecb |
| tag | object | 100.0 | 403 | User-Agent: Python/3.11 aiohttp/3.12.15 |
| date | object | 100.0 | 33 | 2025-08-30 |
| api_key | object | 100.0 | 7 | … |
| model | object | 100.0 | 14 | openai/gpt-oss-120b |
| model_group | object | 100.0 | 23 | gpt-oss-120b |
| custom_llm_provider | object | 100.0 | 4 | together_ai |
| prompt_tokens | int64 | 100.0 | 1611 | 19189 |
| completion_tokens | int64 | 100.0 | 1635 | 4452 |
| cache_read_input_tokens | int64 | 100.0 | 1 | 0 |
| cache_creation_input_tokens | int64 | 100.0 | 1 | 0 |
| spend | float64 | 100.0 | 187 | 0.0055495499999999994 |
| api_requests | int64 | 100.0 | 359 | 21 |
| successful_requests | int64 | 100.0 | 359 | 21 |
| failed_requests | int64 | 100.0 | 1 | 0 |
| created_at | object | 100.0 | 1774 | 2025-08-30 08:08:34.806 |
| updated_at | object | 100.0 | 1781 | 2025-08-30 15:00:23.866 |
| mcp_namespaced_tool_name | object | 100.0 | 1 |  |

### Preview
| id                                   | tag                                     | date       | api_key                                                          | model                        | model_group                   | custom_llm_provider   |   prompt_tokens |   completion_tokens |   cache_read_input_tokens |   cache_creation_input_tokens |      spend |   api_requests |   successful_requests |   failed_requests | created_at              | updated_at              | mcp_namespaced_tool_name   |
|:-------------------------------------|:----------------------------------------|:-----------|:-----------------------------------------------------------------|:-----------------------------|:------------------------------|:----------------------|----------------:|--------------------:|--------------------------:|------------------------------:|-----------:|---------------:|----------------------:|------------------:|:------------------------|:------------------------|:---------------------------|
| cce6c2d8-5f87-4e11-bf5d-c9e0c5518ecb | User-Agent: Python/3.11 aiohttp/3.12.15 | 2025-08-30 | 95400db676f8aaabee0a3c934a5a9ed46d90dfc0c910f9ffd48d6542839115ff | openai/gpt-oss-120b          | gpt-oss-120b                  | together_ai           |           19189 |                4452 |                         0 |                             0 | 0.00554955 |             21 |                    21 |                 0 | 2025-08-30 08:08:34.806 | 2025-08-30 15:00:23.866 |                            |
| 28ba880d-1343-4e02-a1c1-fdb83ecf359a | User-Agent: python-requests             | 2025-08-30 | 95400db676f8aaabee0a3c934a5a9ed46d90dfc0c910f9ffd48d6542839115ff | cohere.embed-multilingual-v3 | cohere-embed-multilingual-v3  | bedrock               |           25122 |                   0 |                         0 |                             0 | 0.0025122  |            152 |                   152 |                 0 | 2025-08-30 13:54:51.353 | 2025-08-30 13:55:02.228 |                            |
| 57bab41f-d09a-4f32-b7b5-9cbc329e0430 | User-Agent: python-requests/2.32.4      | 2025-08-30 | 95400db676f8aaabee0a3c934a5a9ed46d90dfc0c910f9ffd48d6542839115ff | cohere.embed-multilingual-v3 | cohere-embed-multilingual-v3  | bedrock               |           25122 |                   0 |                         0 |                             0 | 0.0025122  |            152 |                   152 |                 0 | 2025-08-30 13:54:51.353 | 2025-08-30 13:55:02.228 |                            |
| a2a77f5b-e9e5-4c37-938b-0417aeabd8c8 | User-Agent: HTTPie                      | 2025-09-04 | 4a7651ea3be2bcc55cff899fb5609fb3614046a8913a30b060f652291cc1797a | apertus-70b-instruct         | swiss-ai/apertus-70b-instruct | openai                |              72 |                 480 |                         0 |                             0 | 0          |              1 |                     1 |                 0 | 2025-09-04 22:56:16.715 | 2025-09-04 22:56:16.715 |                            |
| 096c10f8-e652-49fa-badc-fe02d62ab8c5 | User-Agent: HTTPie/3.2.4                | 2025-09-04 | 4a7651ea3be2bcc55cff899fb5609fb3614046a8913a30b060f652291cc1797a | apertus-70b-instruct         | swiss-ai/apertus-70b-instruct | openai                |              72 |                 480 |                         0 |                             0 | 0          |              1 |                     1 |                 0 | 2025-09-04 22:56:16.715 | 2025-09-04 22:56:16.715 |                            |

## LiteLLM_DailyTeamSpend
*Rows*: **717**  •  *Columns*: **18**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 717 | ca681794-86b9-4e07-9fc8-8d0818dc9483 |
| team_id | object | 100.0 | 3 | 78524c15-3112-4bee-9fb0-ec5ae9453241 |
| date | object | 100.0 | 33 | 2025-08-31 |
| api_key | object | 100.0 | 7 |  |
| model | object | 100.0 | 86 | gpt-oss-120b |
| model_group | object | 100.0 | 29 |  |
| custom_llm_provider | object | 100.0 | 5 |  |
| prompt_tokens | int64 | 100.0 | 363 | 0 |
| completion_tokens | int64 | 100.0 | 330 | 0 |
| spend | float64 | 100.0 | 116 | 0.0 |
| api_requests | int64 | 100.0 | 354 | 6 |
| successful_requests | int64 | 100.0 | 280 | 0 |
| failed_requests | int64 | 100.0 | 119 | 6 |
| created_at | object | 100.0 | 689 | 2025-08-31 03:20:29.119 |
| updated_at | object | 100.0 | 700 | 2025-08-31 20:33:56.371 |
| cache_creation_input_tokens | int64 | 100.0 | 1 | 0 |
| cache_read_input_tokens | int64 | 100.0 | 1 | 0 |
| mcp_namespaced_tool_name | object | 100.0 | 1 |  |

### Preview
| id                                   | team_id                              | date       | api_key                                                          | model                               | model_group             | custom_llm_provider   |   prompt_tokens |   completion_tokens |   spend |   api_requests |   successful_requests |   failed_requests | created_at              | updated_at              |   cache_creation_input_tokens |   cache_read_input_tokens | mcp_namespaced_tool_name   |
|:-------------------------------------|:-------------------------------------|:-----------|:-----------------------------------------------------------------|:------------------------------------|:------------------------|:----------------------|----------------:|--------------------:|--------:|---------------:|----------------------:|------------------:|:------------------------|:------------------------|------------------------------:|--------------------------:|:---------------------------|
| ca681794-86b9-4e07-9fc8-8d0818dc9483 | 78524c15-3112-4bee-9fb0-ec5ae9453241 | 2025-08-31 |                                                                  | gpt-oss-120b                        |                         |                       |               0 |                   0 |       0 |              6 |                     0 |                 6 | 2025-08-31 03:20:29.119 | 2025-08-31 20:33:56.371 |                             0 |                         0 |                            |
| 5750c5b8-7ea1-48c7-8d69-6d1445f295b0 |                                      | 2025-08-30 |                                                                  |                                     |                         |                       |               0 |                   0 |       0 |             17 |                     0 |                17 | 2025-08-30 14:49:30.199 | 2025-08-30 14:53:21.101 |                             0 |                         0 |                            |
| 83e9d7ea-b369-4e00-8d8e-0948fd538e1d |                                      | 2025-08-31 |                                                                  | llama3-2-3b-instruct                |                         |                       |               0 |                   0 |       0 |              7 |                     0 |                 7 | 2025-08-31 01:56:44.551 | 2025-08-31 02:11:19.191 |                             0 |                         0 |                            |
| b3b47eb9-5980-44e0-af94-cb194a5f2a7c | 78524c15-3112-4bee-9fb0-ec5ae9453241 | 2025-09-05 |                                                                  | swissai/apertus-70b-instruct        |                         |                       |               0 |                   0 |       0 |              1 |                     0 |                 1 | 2025-09-05 14:37:28.971 | 2025-09-05 14:37:28.971 |                             0 |                         0 |                            |
| 6ca21167-fb39-46e5-ac55-9a79b408574f |                                      | 2025-08-31 | 95400db676f8aaabee0a3c934a5a9ed46d90dfc0c910f9ffd48d6542839115ff | aisingapore/Gemma-SEA-LION-v3-9B-IT | gemma-sea-lion-v3-9b-it | openai                |             432 |                  42 |       0 |              2 |                     2 |                 0 | 2025-08-31 02:56:36.365 | 2025-08-31 02:58:12.36  |                             0 |                         0 |                            |

## LiteLLM_DailyUserSpend
*Rows*: **704**  •  *Columns*: **18**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 704 | 7c4303ee-8359-4c12-87d6-4f2090631b70 |
| user_id | object | 100.0 | 4 | default_user_id |
| date | object | 100.0 | 33 | 2025-08-31 |
| api_key | object | 100.0 | 7 |  |
| model | object | 100.0 | 86 | apertus-70b-instruct |
| model_group | object | 100.0 | 29 | apertus-70b-instruct |
| custom_llm_provider | object | 100.0 | 5 |  |
| prompt_tokens | int64 | 100.0 | 366 | 0 |
| completion_tokens | int64 | 100.0 | 333 | 0 |
| spend | float64 | 100.0 | 119 | 0.0 |
| created_at | object | 100.0 | 677 | 2025-08-31 13:53:27.578 |
| updated_at | object | 100.0 | 684 | 2025-08-31 23:38:04.416 |
| api_requests | int64 | 100.0 | 357 | 7 |
| failed_requests | int64 | 100.0 | 118 | 7 |
| successful_requests | int64 | 100.0 | 280 | 0 |
| cache_creation_input_tokens | int64 | 100.0 | 1 | 0 |
| cache_read_input_tokens | int64 | 100.0 | 1 | 0 |
| mcp_namespaced_tool_name | object | 100.0 | 1 |  |

### Preview
| id                                   | user_id                              | date       | api_key                                                          | model                   | model_group          | custom_llm_provider   |   prompt_tokens |   completion_tokens |     spend | created_at              | updated_at              |   api_requests |   failed_requests |   successful_requests |   cache_creation_input_tokens |   cache_read_input_tokens | mcp_namespaced_tool_name   |
|:-------------------------------------|:-------------------------------------|:-----------|:-----------------------------------------------------------------|:------------------------|:---------------------|:----------------------|----------------:|--------------------:|----------:|:------------------------|:------------------------|---------------:|------------------:|----------------------:|------------------------------:|--------------------------:|:---------------------------|
| 7c4303ee-8359-4c12-87d6-4f2090631b70 | default_user_id                      | 2025-08-31 |                                                                  | apertus-70b-instruct    | apertus-70b-instruct |                       |               0 |                   0 | 0         | 2025-08-31 13:53:27.578 | 2025-08-31 23:38:04.416 |              7 |                 7 |                     0 |                             0 |                         0 |                            |
| e916b913-22e0-4ec0-9804-cac7bfdce7d0 | bc49bba7-3343-4f03-8601-a4c278228c24 | 2025-08-31 | 95400db676f8aaabee0a3c934a5a9ed46d90dfc0c910f9ffd48d6542839115ff | openai/gpt-oss-120b     | gpt-oss-120b         | together_ai           |            4364 |                1030 | 0.0012726 | 2025-08-31 01:56:15.957 | 2025-08-31 02:13:42.178 |             17 |                 0 |                    17 |                             0 |                         0 |                            |
| 8f02dc44-f6f4-4418-ba9f-a05530ecd240 |                                      | 2025-08-30 |                                                                  |                         |                      |                       |               0 |                   0 | 0         | 2025-08-30 14:49:30.191 | 2025-08-30 14:53:21.095 |             17 |                17 |                     0 |                             0 |                         0 |                            |
| 75937c37-8c20-4acf-8f71-b9ae7b1d7dfe | default_user_id                      | 2025-08-31 |                                                                  | gemma-sea-lion-v3-9b-it |                      |                       |               0 |                   0 | 0         | 2025-08-31 03:20:29.099 | 2025-08-31 03:20:29.099 |              1 |                 1 |                     0 |                             0 |                         0 |                            |
| d0b02521-570b-4ca9-b4d6-8d96b0bf9f7a | f7f2534a-011b-4351-891b-199136ac9a41 | 2025-08-31 | 95400db676f8aaabee0a3c934a5a9ed46d90dfc0c910f9ffd48d6542839115ff | openai/gpt-oss-120b     | gpt-oss-120b         | together_ai           |             203 |                  15 | 3.945e-05 | 2025-08-31 02:11:19.179 | 2025-08-31 02:11:19.179 |              1 |                 0 |                     1 |                             0 |                         0 |                            |

## LiteLLM_EndUserTable
*Rows*: **23805**  •  *Columns*: **7**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| user_id | object | 100.0 | 23805 | enduser_23698314 |
| alias | object | 0.0 | 0 | — |
| spend | float64 | 100.0 | 354 | 0.0011149200000000001 |
| allowed_model_region | object | 0.0 | 0 | — |
| default_model | object | 0.0 | 0 | — |
| budget_id | object | 99.8 | 6 | public_ai_free |
| blocked | bool | 100.0 | 2 | False |

### Preview
| user_id          | alias   |      spend | allowed_model_region   | default_model   | budget_id                            | blocked   |
|:-----------------|:--------|-----------:|:-----------------------|:----------------|:-------------------------------------|:----------|
| enduser_23698314 |         | 0.00111492 |                        |                 | public_ai_free                       | False     |
| enduser_9d6a4807 |         | 0.0008196  |                        |                 | Public AI Free Tier                  | False     |
| enduser_54500eeb |         | 0.0005595  |                        |                 |                                      | False     |
| enduser_3a9a9d4d |         | 0.00616797 |                        |                 | public_ai_free                       | False     |
| enduser_8f18a31e |         | 0          |                        |                 | bc49bba7-3343-4f03-8601-a4c278228c24 | False     |

## LiteLLM_GuardrailsTable
*Rows*: **0**  •  *Columns*: **6**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_HealthCheckTable
*Rows*: **0**  •  *Columns*: **13**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_MCPServerTable
*Rows*: **0**  •  *Columns*: **19**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_ManagedFileTable
*Rows*: **0**  •  *Columns*: **9**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_ManagedObjectTable
*Rows*: **0**  •  *Columns*: **10**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_ManagedVectorStoresTable
*Rows*: **0**  •  *Columns*: **9**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_ModelTable
*Rows*: **0**  •  *Columns*: **6**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_ObjectPermissionTable
*Rows*: **1**  •  *Columns*: **4**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| object_permission_id | object | 100.0 | 1 | 10c1571c-b409-418d-86c7-a2c0bf3e5be0 |
| mcp_servers | object | 100.0 | 1 | {} |
| vector_stores | object | 100.0 | 1 | {} |
| mcp_access_groups | object | 100.0 | 1 | {} |

### Preview
| object_permission_id                 | mcp_servers   | vector_stores   | mcp_access_groups   |
|:-------------------------------------|:--------------|:----------------|:--------------------|
| 10c1571c-b409-418d-86c7-a2c0bf3e5be0 | {}            | {}              | {}                  |

## LiteLLM_OrganizationMembership
*Rows*: **0**  •  *Columns*: **7**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_OrganizationTable
*Rows*: **0**  •  *Columns*: **12**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_PromptTable
*Rows*: **0**  •  *Columns*: **6**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_ProxyModelTable
*Rows*: **0**  •  *Columns*: **8**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_SpendLogs
*Rows*: **713244**  •  *Columns*: **29**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| request_id | object | 100.0 | 713244 | chatcmpl-5fb2a0fb885548438d556208a15fa445 |
| call_type | object | 100.0 | 4 | acompletion |
| api_key | object | 0.0 | 0 | — |
| spend | float64 | 100.0 | 15067 | 0.0 |
| total_tokens | int32 | 100.0 | 20486 | 2282 |
| prompt_tokens | int32 | 100.0 | 18663 | 1250 |
| completion_tokens | int32 | 100.0 | 5835 | 1032 |
| startTime | object | 100.0 | 712789 | 2025-09-05 14:49:39.518 |
| endTime | object | 100.0 | 713019 | 2025-09-05 14:50:02.744 |
| completionStartTime | object | 100.0 | 712781 | 2025-09-05 14:49:39.715 |
| model | object | 100.0 | 86 | apertus-70b-instruct |
| model_id | object | 100.0 | 57 | … |
| model_group | object | 100.0 | 29 | swiss-ai/apertus-70b-instruct |
| custom_llm_provider | object | 100.0 | 5 | openai |
| api_base | object | 100.0 | 21 | http://llm-services-router-service.llm-… |
| user | object | 100.0 | 4 | user_a9ec9a5b |
| metadata | object | 0.0 | 0 | — |
| cache_hit | object | 100.0 | 2 | False |
| cache_key | object | 100.0 | 44 | Cache OFF |
| request_tags | object | 100.0 | 311 | ["User-Agent: Python", "User-Agent: Python/3.11… |
| team_id | object | 100.0 | 3 | 78524c15-3112-4bee-9fb0-ec5ae9453241 |
| end_user | object | 100.0 | 21267 | enduser_b882f583 |
| requester_ip_address | object | 0.0 | 0 | — |
| messages | object | 0.0 | 0 | — |
| response | object | 0.0 | 0 | — |
| proxy_server_request | object | 0.0 | 0 | — |
| session_id | object | 100.0 | 713238 | 2cd19994-7bff-4744-853d-b82b04afd981 |
| status | object | 100.0 | 2 | success |
| mcp_namespaced_tool_name | object | 0.0 | 0 | — |

### Preview
| request_id                                    | call_type   | api_key   |      spend |   total_tokens |   prompt_tokens |   completion_tokens | startTime               | endTime                 | completionStartTime     | model                             | model_id                                                         | model_group                      | custom_llm_provider   | api_base                                                             | user          | metadata   | cache_hit   | cache_key   | request_tags                                                      | team_id                              | end_user         | requester_ip_address   | messages   | response   | proxy_server_request   | session_id                           | status   | mcp_namespaced_tool_name   |
|:----------------------------------------------|:------------|:----------|-----------:|---------------:|----------------:|--------------------:|:------------------------|:------------------------|:------------------------|:----------------------------------|:-----------------------------------------------------------------|:---------------------------------|:----------------------|:---------------------------------------------------------------------|:--------------|:-----------|:------------|:------------|:------------------------------------------------------------------|:-------------------------------------|:-----------------|:-----------------------|:-----------|:-----------|:-----------------------|:-------------------------------------|:---------|:---------------------------|
| chatcmpl-5fb2a0fb885548438d556208a15fa445     | acompletion |           | 0          |           2282 |            1250 |                1032 | 2025-09-05 14:49:39.518 | 2025-09-05 14:50:02.744 | 2025-09-05 14:49:39.715 | apertus-70b-instruct              | 7b5ca22d59fc17e990c114ffe63ef5257aec66675d21e357b7448e82c306b1f9 | swiss-ai/apertus-70b-instruct    | openai                | http://llm-services-router-service.llm-services.svc.cluster.local/v1 | user_a9ec9a5b |            | False       | Cache OFF   | ["User-Agent: Python", "User-Agent: Python/3.11 aiohttp/3.12.15"] | 78524c15-3112-4bee-9fb0-ec5ae9453241 | enduser_b882f583 |                        |            |            |                        | 2cd19994-7bff-4744-853d-b82b04afd981 | success  |                            |
| chatcmpl-ebb9dd4d65b147378887029331823dc6     | acompletion |           | 0          |           1035 |            1008 |                  27 | 2025-09-05 14:50:09.408 | 2025-09-05 14:50:09.612 | 2025-09-05 14:50:09.463 | apertus-8b-instruct               | 2a1d1aeac795bd4c010b509c17a7725ca981fc29e52270ac761cf5167f415ecf | swiss-ai/apertus-8b-instruct     | openai                | http://llm-services-router-service.llm-services.svc.cluster.local/v1 | user_a9ec9a5b |            | False       | Cache OFF   | ["User-Agent: dp", "User-Agent: dp/JS 5.16.0"]                    | 78524c15-3112-4bee-9fb0-ec5ae9453241 | enduser_d41d8cd9 |                        |            |            |                        | e52f84a8-75e3-44e0-8915-b4dec532056a | success  |                            |
| chatcmpl-004f421a78b94ea1aa7687a73d01c0cb     | acompletion |           | 0          |           2131 |            1050 |                1081 | 2025-09-05 14:50:06.85  | 2025-09-05 14:50:12.833 | 2025-09-05 14:50:06.901 | apertus-8b-instruct               | 2a1d1aeac795bd4c010b509c17a7725ca981fc29e52270ac761cf5167f415ecf | swiss-ai/apertus-8b-instruct     | openai                | http://llm-services-router-service.llm-services.svc.cluster.local/v1 | user_a9ec9a5b |            | False       | Cache OFF   | ["User-Agent: dp", "User-Agent: dp/JS 5.16.0"]                    | 78524c15-3112-4bee-9fb0-ec5ae9453241 | enduser_d41d8cd9 |                        |            |            |                        | 141ab921-6401-434e-8e50-02ac6f2cb9a9 | success  |                            |
| chatcmpl-22c421e416594538a888b99a733df35f     | acompletion |           | 0          |           1591 |             781 |                 810 | 2025-09-05 14:49:55.748 | 2025-09-05 14:50:15.394 | 2025-09-05 14:49:55.906 | apertus-70b-instruct              | 7b5ca22d59fc17e990c114ffe63ef5257aec66675d21e357b7448e82c306b1f9 | swiss-ai/apertus-70b-instruct    | openai                | http://llm-services-router-service.llm-services.svc.cluster.local/v1 | user_a9ec9a5b |            | False       | Cache OFF   | ["User-Agent: Python", "User-Agent: Python/3.11 aiohttp/3.12.15"] | 78524c15-3112-4bee-9fb0-ec5ae9453241 | enduser_edab754c |                        |            |            |                        | b5b3d93f-853e-46c3-bc99-1bc50f1a089f | success  |                            |
| chatcmpl-eccfc6ae-fc8b-4d8a-bdfe-50b05dc63230 | acompletion |           | 0.00021356 |           1124 |            1066 |                  58 | 2025-09-05 14:50:15.481 | 2025-09-05 14:50:16.445 | 2025-09-05 14:50:16.445 | eu.meta.llama3-2-3b-instruct-v1:0 | b83416aefaf4f57d51ba88c32fd193b9ad190fff6a9e6ee415483bb5f6ebb8f2 | meta-llama/Llama-3.2-3B-Instruct | bedrock               | https://bedrock-runtime.eu-…                                         | user_a9ec9a5b |            | None        | Cache OFF   | ["User-Agent: Python", "User-Agent: Python/3.11 aiohttp/3.12.15"] | 78524c15-3112-4bee-9fb0-ec5ae9453241 | enduser_edab754c |                        |            |            |                        | 9ac4a167-a35d-4667-95f0-d272b7235f29 | success  |                            |

## LiteLLM_TeamMembership
*Rows*: **0**  •  *Columns*: **4**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_TeamTable
*Rows*: **1**  •  *Columns*: **23**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| team_id | object | 100.0 | 1 | 78524c15-3112-4bee-9fb0-ec5ae9453241 |
| team_alias | object | 100.0 | 1 | Open WebUI |
| organization_id | object | 0.0 | 0 | — |
| admins | object | 100.0 | 1 | {} |
| members | object | 100.0 | 1 | {} |
| members_with_roles | object | 100.0 | 1 | [{"role": "admin", "user_id": "default_user_id",… |
| metadata | object | 100.0 | 1 | {"logging": [], "guardrails": []} |
| max_budget | float64 | 0.0 | 0 | — |
| spend | float64 | 100.0 | 1 | 95.18300827000034 |
| models | object | 100.0 | 1 | {all-proxy-models} |
| max_parallel_requests | float64 | 0.0 | 0 | — |
| tpm_limit | float64 | 0.0 | 0 | — |
| rpm_limit | float64 | 0.0 | 0 | — |
| budget_duration | object | 0.0 | 0 | — |
| budget_reset_at | object | 0.0 | 0 | — |
| blocked | bool | 100.0 | 1 | False |
| created_at | object | 100.0 | 1 | 2025-08-31 03:00:30.757 |
| updated_at | object | 100.0 | 1 | 2025-10-01 03:34:57.976 |
| model_spend | object | 100.0 | 1 | {} |
| model_max_budget | object | 100.0 | 1 | {} |
| model_id | float64 | 0.0 | 0 | — |
| team_member_permissions | object | 100.0 | 1 | {} |
| object_permission_id | object | 0.0 | 0 | — |

### Preview
| team_id                              | team_alias   | organization_id   | admins   | members   | members_with_roles                                                              | metadata                          |   max_budget |   spend | models             |   max_parallel_requests |   tpm_limit |   rpm_limit | budget_duration   | budget_reset_at   | blocked   | created_at              | updated_at              | model_spend   | model_max_budget   |   model_id | team_member_permissions   | object_permission_id   |
|:-------------------------------------|:-------------|:------------------|:---------|:----------|:--------------------------------------------------------------------------------|:----------------------------------|-------------:|--------:|:-------------------|------------------------:|------------:|------------:|:------------------|:------------------|:----------|:------------------------|:------------------------|:--------------|:-------------------|-----------:|:--------------------------|:-----------------------|
| 78524c15-3112-4bee-9fb0-ec5ae9453241 | Open WebUI   |                   | {}       | {}        | [{"role": "admin", "user_id": "default_user_id", "user_email": null}, {"role":… | {"logging": [], "guardrails": []} |          nan |  95.183 | {all-proxy-models} |                     nan |         nan |         nan |                   |                   | False     | 2025-08-31 03:00:30.757 | 2025-10-01 03:34:57.976 | {}            | {}                 |        nan | {}                        |                        |

## LiteLLM_UserNotifications
*Rows*: **0**  •  *Columns*: **5**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| — | — | 0 | 0 | — |

### Preview
_No rows._

## LiteLLM_UserTable
*Rows*: **4**  •  *Columns*: **24**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| user_id | object | 100.0 | 4 | be66e41e-b2d7-45f6-ab43-e79785b034aa |
| user_alias | object | 0.0 | 0 | — |
| team_id | object | 75.0 | 2 | 78524c15-3112-4bee-9fb0-ec5ae9453241 |
| sso_user_id | object | 0.0 | 0 | — |
| organization_id | object | 0.0 | 0 | — |
| password | object | 0.0 | 0 | — |
| teams | object | 100.0 | 2 | {78524c15-3112-4bee-9fb0-ec5ae9453241} |
| user_role | object | 100.0 | 3 | proxy_admin_viewer |
| max_budget | float64 | 0.0 | 0 | — |
| spend | float64 | 100.0 | 2 | 0.0 |
| user_email | object | 75.0 | 3 | email_0d99935a |
| models | object | 75.0 | 1 | {no-default-models} |
| metadata | object | 0.0 | 0 | — |
| max_parallel_requests | float64 | 0.0 | 0 | — |
| tpm_limit | float64 | 0.0 | 0 | — |
| rpm_limit | float64 | 0.0 | 0 | — |
| budget_duration | object | 0.0 | 0 | — |
| budget_reset_at | object | 0.0 | 0 | — |
| allowed_cache_controls | object | 100.0 | 1 | {} |
| model_spend | object | 100.0 | 1 | {} |
| model_max_budget | object | 100.0 | 1 | {} |
| created_at | object | 100.0 | 4 | 2025-09-13 16:24:55.742 |
| updated_at | object | 100.0 | 4 | 2025-09-16 08:35:13.055 |
| object_permission_id | object | 0.0 | 0 | — |

### Preview
| user_id                              | user_alias   | team_id                              | sso_user_id   | organization_id   | password   | teams                                  | user_role          |   max_budget |   spend | user_email     | models              | metadata   |   max_parallel_requests |   tpm_limit |   rpm_limit | budget_duration   | budget_reset_at   | allowed_cache_controls   | model_spend   | model_max_budget   | created_at              | updated_at              | object_permission_id   |
|:-------------------------------------|:-------------|:-------------------------------------|:--------------|:------------------|:-----------|:---------------------------------------|:-------------------|-------------:|--------:|:---------------|:--------------------|:-----------|------------------------:|------------:|------------:|:------------------|:------------------|:-------------------------|:--------------|:-------------------|:------------------------|:------------------------|:-----------------------|
| be66e41e-b2d7-45f6-ab43-e79785b034aa |              | 78524c15-3112-4bee-9fb0-ec5ae9453241 |               |                   |            | {78524c15-3112-4bee-9fb0-ec5ae9453241} | proxy_admin_viewer |          nan |  0      | email_0d99935a | {no-default-models} |            |                     nan |         nan |         nan |                   |                   | {}                       | {}            | {}                 | 2025-09-13 16:24:55.742 | 2025-09-16 08:35:13.055 |                        |
| 5340d3bd-597e-4601-b63c-57b51769d1ae |              | 8aff44bc-3d72-4ae0-8d94-02937ffd516d |               |                   |            | {}                                     | internal_user      |          nan |  0      | email_2dda535f | {no-default-models} |            |                     nan |         nan |         nan |                   |                   | {}                       | {}            | {}                 | 2025-08-31 14:40:50.094 | 2025-09-04 07:30:27.816 |                        |
| 2af338e6-7254-49c5-aa34-a5435a3e5b55 |              | 78524c15-3112-4bee-9fb0-ec5ae9453241 |               |                   |            | {78524c15-3112-4bee-9fb0-ec5ae9453241} | proxy_admin_viewer |          nan |  0      | email_ae977c3d | {no-default-models} |            |                     nan |         nan |         nan |                   |                   | {}                       | {}            | {}                 | 2025-09-13 16:22:43.309 | 2025-09-13 16:22:57.856 |                        |
| default_user_id                      |              |                                      |               |                   |            | {78524c15-3112-4bee-9fb0-ec5ae9453241} | proxy_admin        |          nan | 96.9894 |                |                     |            |                     nan |         nan |         nan |                   |                   | {}                       | {}            | {}                 | 2025-08-30 04:04:23.382 | 2025-10-01 03:34:57.957 |                        |

## LiteLLM_VerificationToken
*Rows*: **65**  •  *Columns*: **31**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| token | object | 100.0 | 65 | … |
| key_name | object | 100.0 | 65 | sk-...tT0g |
| key_alias | object | 6.2 | 4 | OpenWebUI Master |
| soft_budget_cooldown | bool | 100.0 | 1 | False |
| spend | float64 | 100.0 | 4 | 0.0 |
| expires | object | 93.8 | 54 | 2025-08-31 04:04:23.394 |
| models | object | 100.0 | 4 | {} |
| aliases | object | 100.0 | 1 | {} |
| config | object | 100.0 | 1 | {} |
| user_id | object | 100.0 | 4 | default_user_id |
| team_id | object | 100.0 | 2 | litellm-dashboard |
| permissions | object | 100.0 | 1 | {} |
| max_parallel_requests | float64 | 0.0 | 0 | — |
| metadata | object | 100.0 | 2 | {} |
| blocked | object | 0.0 | 0 | — |
| tpm_limit | float64 | 0.0 | 0 | — |
| rpm_limit | float64 | 0.0 | 0 | — |
| max_budget | float64 | 93.8 | 1 | 10.0 |
| budget_duration | object | 0.0 | 0 | — |
| budget_reset_at | object | 0.0 | 0 | — |
| allowed_cache_controls | object | 100.0 | 1 | {} |
| model_spend | object | 100.0 | 1 | {} |
| model_max_budget | object | 100.0 | 1 | {} |
| budget_id | object | 0.0 | 0 | — |
| organization_id | object | 0.0 | 0 | — |
| created_at | object | 100.0 | 65 | 2025-08-30 04:04:23.399 |
| created_by | object | 6.2 | 1 | default_user_id |
| updated_at | object | 100.0 | 65 | 2025-08-30 04:04:23.399 |
| updated_by | object | 6.2 | 1 | default_user_id |
| allowed_routes | object | 100.0 | 2 | {} |
| object_permission_id | object | 1.5 | 1 | 10c1571c-b409-418d-86c7-a2c0bf3e5be0 |

### Preview
| token                                                            | key_name   | key_alias   | soft_budget_cooldown   |   spend | expires                 | models              | aliases   | config   | user_id                              | team_id           | permissions   |   max_parallel_requests | metadata   | blocked   |   tpm_limit |   rpm_limit |   max_budget | budget_duration   | budget_reset_at   | allowed_cache_controls   | model_spend   | model_max_budget   | budget_id   | organization_id   | created_at              | created_by   | updated_at              | updated_by   | allowed_routes   |
|:-----------------------------------------------------------------|:-----------|:------------|:-----------------------|--------:|:------------------------|:--------------------|:----------|:---------|:-------------------------------------|:------------------|:--------------|------------------------:|:-----------|:----------|------------:|------------:|-------------:|:------------------|:------------------|:-------------------------|:--------------|:-------------------|:------------|:------------------|:------------------------|:-------------|:------------------------|:-------------|:-----------------|
| 38b5148c3bb2578e3120f4b4f7df62e62d94813e5209b19a2ad2ff03882d3b1f | sk-...tT0g |             | False                  |       0 | 2025-08-31 04:04:23.394 | {}                  | {}        | {}       | default_user_id                      | litellm-dashboard | {}            |                     nan | {}         |           |         nan |         nan |           10 |                   |                   | {}                       | {}            | {}                 |             |                   | 2025-08-30 04:04:23.399 |              | 2025-08-30 04:04:23.399 |              | {}               |
| 7b1b496ad6ba2bf6229ec58d189df54efc90d53de0e97f28de22927f0750f760 | sk-...7bSw |             | False                  |       0 | 2025-08-31 07:13:27.355 | {}                  | {}        | {}       | default_user_id                      | litellm-dashboard | {}            |                     nan | {}         |           |         nan |         nan |           10 |                   |                   | {}                       | {}            | {}                 |             |                   | 2025-08-30 07:13:27.37  |              | 2025-08-30 07:13:27.37  |              | {}               |
| 2c68351df3f9ea83e87a39a4f33c046fbc3bc67896cfd530c239b554038c3c48 | sk-...0LxQ |             | False                  |       0 | 2025-09-25 00:00:00     | {no-default-models} | {}        | {}       | be66e41e-b2d7-45f6-ab43-e79785b034aa | litellm-dashboard | {}            |                     nan | {}         |           |         nan |         nan |           10 |                   |                   | {}                       | {}            | {}                 |             |                   | 2025-09-24 06:20:21.785 |              | 2025-09-24 06:20:21.785 |              | {}               |
| 15732a8151dfe014107d5e446a6848f39809432099c21d728fce83afe5b978cc | sk-...IYfg |             | False                  |       0 | 2025-09-27 00:00:00     | {no-default-models} | {}        | {}       | be66e41e-b2d7-45f6-ab43-e79785b034aa | litellm-dashboard | {}            |                     nan | {}         |           |         nan |         nan |           10 |                   |                   | {}                       | {}            | {}                 |             |                   | 2025-09-26 14:09:24.381 |              | 2025-09-26 14:09:24.381 |              | {}               |
| 110df83905b2ae9e1b0119306ec3442578316c870ab15a95307c9c6c46f7f2cc | sk-...Ypxg |             | False                  |       0 | 2025-09-01 02:54:59.315 | {}                  | {}        | {}       | default_user_id                      | litellm-dashboard | {}            |                     nan | {}         |           |         nan |         nan |           10 |                   |                   | {}                       | {}            | {}                 |             |                   | 2025-08-31 02:54:59.322 |              | 2025-08-31 02:57:22.503 |              | {}               |

## _prisma_migrations
*Rows*: **35**  •  *Columns*: **8**

### Schema
| column | dtype | non-null % | unique | example |
|:--|:--|--:|--:|:--|
| id | object | 100.0 | 35 | df17cff0-657b-43ab-a0d1-1b402e9a3034 |
| checksum | object | 100.0 | 35 | … |
| finished_at | object | 100.0 | 35 | 2025-08-30 03:51:34.361161+00 |
| migration_name | object | 100.0 | 35 | 20250326162113_baseline |
| logs | object | 0.0 | 0 | — |
| rolled_back_at | object | 0.0 | 0 | — |
| started_at | object | 100.0 | 35 | 2025-08-30 03:51:34.116635+00 |
| applied_steps_count | int32 | 100.0 | 1 | 1 |

### Preview
| id                                   | checksum                                                         | finished_at                   | migration_name                                         | logs   | rolled_back_at   | started_at                    |   applied_steps_count |
|:-------------------------------------|:-----------------------------------------------------------------|:------------------------------|:-------------------------------------------------------|:-------|:-----------------|:------------------------------|----------------------:|
| df17cff0-657b-43ab-a0d1-1b402e9a3034 | f5a4569816cb7fb1166c231789f0e5e732fd6c0c34b8f5c9badd405cd2ab4931 | 2025-08-30 03:51:34.361161+00 | 20250326162113_baseline                                |        |                  | 2025-08-30 03:51:34.116635+00 |                     1 |
| aeaa6779-89f4-4b83-8aa1-3daa1119048f | 758f9d3422e8b169973658b607dc2a5f8859a69decd032f723010fb534c2d77e | 2025-08-30 03:51:34.390362+00 | 20250326171002_add_daily_user_table                    |        |                  | 2025-08-30 03:51:34.362861+00 |                     1 |
| 83a2fb02-7dfc-4b20-9ac2-dc50ea8105c9 | ed0d84d0e48ae306578bf19e351f6a17e49b421d4e1ef246f80fb4da74310605 | 2025-09-29 13:14:25.173043+00 | 20250918083359_drop_spec_version_column_from_mcp_table |        |                  | 2025-09-29 13:14:25.083291+00 |                     1 |
| 23e5d1a8-a4cc-4945-b329-1f8c0fd7c67f | a2be53684807e1c1f0479903bd58d9bdc023d4ec334f18eac0efa5c577b71579 | 2025-08-30 03:51:34.398455+00 | 20250327180120_add_api_requests_to_daily_user_table    |        |                  | 2025-08-30 03:51:34.392037+00 |                     1 |
| 56d7c360-ea4b-45bb-ac81-41aeec49122a | 799343c2bccab56157913a8e29bf494b8873ba6d5d23e5535ef32ace1af27f17 | 2025-08-30 03:51:34.420669+00 | 20250329084805_new_cron_job_table                      |        |                  | 2025-08-30 03:51:34.400044+00 |                     1 |


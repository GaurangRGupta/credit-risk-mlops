# Product Backlog: Credit Risk Classifier

## Scrum Roles
- **Product Owner:** defines and prioritises user stories (Gaurang Gupta)
- **Scrum Master:** removes blockers and runs the ceremonies
- **Development Team:** builds, tests and deploys the solution

## User Stories
| ID | User Story | Priority | Sprint |
|----|-----------|----------|--------|
| US1 | As a bank analyst, I want to predict credit risk from applicant data so I can approve loans safely. | High | 1 |
| US2 | As a data scientist, I want experiments tracked so I can pick the best model. | High | 1 |
| US3 | As a developer, I want predictions served through a REST API and web UI. | High | 2 |
| US4 | As a team, we want automated testing and deployment (CI/CD) so broken code never reaches main. | Medium | 2 |
| US5 | As an ops engineer, I want to monitor API traffic, errors and latency. | Medium | 3 |

## Sprint Plan
- **Sprint 1:** ETL pipeline, feature store, model training with MLflow (US1, US2)
- **Sprint 2:** Flask API, UI, tests, GitHub Actions CI, Docker (US3, US4)
- **Sprint 3:** Kubernetes, Azure deployment, Prometheus + Grafana monitoring (US5)

## Definition of Done
- Code reviewed through a pull request and merged to main
- CI pipeline green (lint, train, tests, Docker build)
- Model meets the quality gate (accuracy ≥ 0.70)

## Sprint Retrospective
- **Went well:** feature-branch workflow, CI caught issues early, MLflow made model comparison easy.
- **To improve:** a commit landed on main by mistake (fixed with cherry-pick); Azure port mismatch (80 vs 5000) cost time.
- **Next:** add a CD workflow to Azure and data drift monitoring.

# Requirements Document

## Introduction

CopyAir es una plataforma SaaS cloud-native para entrenamiento e inferencia de modelos de IA especializados en VFX y deaging. La plataforma permite a usuarios subir datasets, configurar entrenamientos, monitorear progreso en tiempo real, y realizar inferencia con modelos entrenados, todo a través de una interfaz web moderna con estética de software profesional VFX.

## Glossary

- **CopyAir_Platform**: El sistema completo SaaS incluyendo frontend, backend APIs, y orquestación de infraestructura
- **Training_Job**: Una instancia de entrenamiento de modelo ejecutándose en infraestructura remota
- **Dataset_Pair**: Conjunto de imágenes input y ground truth sincronizadas para entrenamiento
- **Vast_Instance**: Máquina virtual alquilada en Vast.ai para cómputo GPU
- **AWS_Orchestrator**: Servicios AWS (Step Functions, Lambda) que coordinan el ciclo de vida de jobs
- **Model_Artifact**: Archivo .pt/.pth resultante del entrenamiento
- **Inference_Job**: Proceso de aplicar un modelo entrenado a nuevas imágenes/videos
- **Credit_System**: Sistema de facturación basado en créditos para uso de recursos

## Requirements

### Requirement 1: User Authentication and Account Management

**User Story:** Como usuario, quiero crear una cuenta y gestionar mi perfil, para acceder a la plataforma de forma segura y personalizada.

#### Acceptance Criteria

1. WHEN a new user registers, THE CopyAir_Platform SHALL create an account with email verification
2. WHEN a user logs in, THE AWS_Orchestrator SHALL authenticate using AWS Cognito
3. WHEN a user accesses their profile, THE CopyAir_Platform SHALL display current credit balance and usage history
4. THE Credit_System SHALL track and display remaining credits in real-time
5. WHEN credits are low (< 10%), THE CopyAir_Platform SHALL notify the user

### Requirement 2: Dataset Upload and Validation

**User Story:** Como usuario, quiero subir datasets de input y ground truth, para entrenar modelos personalizados con mis datos.

#### Acceptance Criteria

1. WHEN a user uploads folders, THE CopyAir_Platform SHALL accept paired input/ground truth image sequences
2. WHEN images are uploaded, THE CopyAir_Platform SHALL validate format compatibility (PNG, JPG, EXR)
3. WHEN validation completes, THE CopyAir_Platform SHALL display synchronized carousel viewer
4. THE Dataset_Pair SHALL maintain frame-by-frame correspondence between input and ground truth
5. WHEN scrolling the carousel, THE CopyAir_Platform SHALL keep both strips perfectly synchronized
6. WHEN selecting validation frames, THE CopyAir_Platform SHALL limit selection to maximum 4 frames
7. THE CopyAir_Platform SHALL store uploaded data with 7-day retention policy

### Requirement 3: AI-Powered Dataset Analysis

**User Story:** Como usuario, quiero que la IA analice mi dataset automáticamente, para identificar problemas de calidad antes del entrenamiento.

#### Acceptance Criteria

1. WHEN user clicks "Analyze Consistency", THE CopyAir_Platform SHALL run automated quality checks
2. WHEN analysis completes, THE CopyAir_Platform SHALL generate a Health Score (0-100)
3. WHEN issues are detected, THE CopyAir_Platform SHALL list specific warnings with frame numbers
4. WHEN clicking a warning, THE CopyAir_Platform SHALL jump the carousel to the problematic frame
5. THE CopyAir_Platform SHALL detect common issues: jitter, exposure changes, misalignment

### Requirement 4: Training Configuration and Cost Estimation

**User Story:** Como usuario, quiero configurar parámetros de entrenamiento y ver costos estimados, para tomar decisiones informadas sobre mi presupuesto.

#### Acceptance Criteria

1. THE CopyAir_Platform SHALL provide 4 preset configurations: Fast Draft, Standard, High Fidelity, Experimental
2. WHEN selecting a preset, THE CopyAir_Platform SHALL display real-time cost estimation
3. WHEN enabling multi-experiment mode, THE CopyAir_Platform SHALL allow parallel job configurations
4. THE Credit_System SHALL calculate costs based on: GPU hours, data transfer, storage usage
5. WHEN insufficient credits, THE CopyAir_Platform SHALL prevent job submission and suggest top-up

### Requirement 5: Infrastructure Orchestration and Job Management

**User Story:** Como administrador del sistema, quiero que la plataforma gestione automáticamente la infraestructura, para optimizar costos y disponibilidad.

#### Acceptance Criteria

1. WHEN a training job starts, THE AWS_Orchestrator SHALL provision Vast_Instance automatically
2. WHEN provisioning, THE AWS_Orchestrator SHALL select optimal GPU configuration based on job requirements
3. WHEN job completes, THE AWS_Orchestrator SHALL terminate Vast_Instance within 5 minutes
4. THE AWS_Orchestrator SHALL handle job failures and retry logic (max 3 attempts)
5. WHEN instance becomes unavailable, THE AWS_Orchestrator SHALL migrate job to alternative provider
6. THE AWS_Orchestrator SHALL monitor job health and send status updates every 30 seconds

### Requirement 6: Real-time Training Monitoring

**User Story:** Como usuario, quiero monitorear el progreso de entrenamiento en tiempo real, para evaluar la calidad del modelo durante el proceso.

#### Acceptance Criteria

1. WHEN training starts, THE CopyAir_Platform SHALL display provisioning status with engaging messages
2. WHEN GPU is active, THE CopyAir_Platform SHALL show real-time loss charts with log scale option
3. WHEN training progresses, THE CopyAir_Platform SHALL update preview images every 5 seconds maximum
4. THE CopyAir_Platform SHALL display clear notification that training runs server-side
5. WHEN user closes browser, THE Training_Job SHALL continue running uninterrupted
6. THE CopyAir_Platform SHALL provide "Stop & Save" and "Kill Process" controls

### Requirement 7: Model Download and Artifact Management

**User Story:** Como usuario, quiero descargar mis modelos entrenados, para usar localmente o en otros sistemas.

#### Acceptance Criteria

1. WHEN training completes successfully, THE CopyAir_Platform SHALL generate downloadable Model_Artifact
2. THE Model_Artifact SHALL be available in standard PyTorch formats (.pt, .pth)
3. WHEN 7-day retention expires, THE CopyAir_Platform SHALL send deletion warnings 24 hours prior
4. THE CopyAir_Platform SHALL provide model metadata: training parameters, final loss, validation scores
5. WHEN downloading, THE CopyAir_Platform SHALL verify file integrity with checksums

### Requirement 8: Tiered Inference System

**User Story:** Como usuario, quiero realizar inferencia con diferentes niveles de servicio, para balancear costo y calidad según mis necesidades.

#### Acceptance Criteria

1. THE CopyAir_Platform SHALL provide Free Tier inference with 100 frame limit and 200MB max file size
2. THE CopyAir_Platform SHALL provide Pro Tier inference supporting up to 8K resolution and 8GB files
3. WHEN using Pro Tier, THE Credit_System SHALL charge $0.20 per 240 frames dynamically
4. THE CopyAir_Platform SHALL display before/after comparison with slider video player
5. WHEN uploading for inference, THE CopyAir_Platform SHALL validate file format and size limits
6. THE Inference_Job SHALL process video sequences maintaining temporal consistency

### Requirement 9: API Architecture and Integration

**User Story:** Como desarrollador, quiero APIs bien documentadas y consistentes, para integrar CopyAir con otros sistemas.

#### Acceptance Criteria

1. THE CopyAir_Platform SHALL provide RESTful APIs for all core operations
2. THE CopyAir_Platform SHALL implement GraphQL endpoint for complex queries
3. WHEN API calls are made, THE CopyAir_Platform SHALL return consistent error formats
4. THE CopyAir_Platform SHALL provide OpenAPI/Swagger documentation
5. THE CopyAir_Platform SHALL implement rate limiting: 1000 requests/hour for authenticated users
6. THE CopyAir_Platform SHALL support webhook notifications for job status changes

### Requirement 10: Security and Compliance

**User Story:** Como usuario empresarial, quiero garantías de seguridad y privacidad de datos, para cumplir con políticas corporativas.

#### Acceptance Criteria

1. THE CopyAir_Platform SHALL encrypt all data in transit using TLS 1.3
2. THE CopyAir_Platform SHALL encrypt all data at rest using AES-256
3. WHEN processing data, THE Vast_Instance SHALL use isolated containers with no data persistence
4. THE CopyAir_Platform SHALL implement GDPR-compliant data deletion
5. THE AWS_Orchestrator SHALL log all access and operations for audit trails
6. THE CopyAir_Platform SHALL provide data processing agreements for enterprise customers
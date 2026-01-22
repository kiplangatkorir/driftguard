# Changelog

All notable changes to DriftGuard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-01-22

### Added
- **Adaptive Thread Pool Sizing** (#13)
  - Dynamic thread pool sizing based on CPU cores and workload
  - Configurable max workers via `DriftConfig`
  - Auto-scaling workers for optimal performance
  - Improved scalability for large datasets

- **Batch Feature Processing** (#14)
  - Features grouped by type (categorical vs continuous)
  - Parallel batch processing for efficiency
  - Reduced processing overhead
  - Better cache utilization

- **AlertManager Integration with ModelMonitor** (#9)
  - Automatic email alerts on performance degradation
  - Detailed alert messages with metric information
  - Configurable alert thresholds
  - Integration with existing alert infrastructure

### Improved
- Performance optimization for drift detection
- Better resource utilization with adaptive threading
- Enhanced monitoring capabilities with integrated alerts

### Fixed
- Thread pool performance bottlenecks
- Feature processing inefficiencies

## [0.1.5] - 2025-06-11

### Added
- **Automated Reporting**
  - PDF report generation
  - Email integration for reports
  - Enhanced content with performance metrics

- **Visualization**
  - Drift score charts
  - Importance change tracking
  - Printable report format

- **Monitoring Enhancements**
  - Performance thresholds
  - Parallel processing with ThreadPoolExecutor
  - SHAP-based feature importance tracking

- **Alerting Improvements**
  - Rate limiting for alerts
  - Detailed drift score alerts
  - PDF attachments with alerts

### Improved
- Better performance tracking
- Enhanced drift detection accuracy

## [0.1.0] - 2025-02-06

### Added
- Initial release of DriftGuard
- Data drift detection (KS test, PSI, JSD, Wasserstein distance)
- Concept drift monitoring
- Model performance tracking
- Email alert system
- Basic reporting capabilities
- FastAPI integration support
- CLI interface

---

For more information, visit [DriftGuard on GitHub](https://github.com/kiplangatkorir/driftguard)
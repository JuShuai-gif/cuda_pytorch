// Chapter 11: Logger implementation stub.
// Logger is primarily header-only (inline). This file exists for build system
// compatibility and future expansion (e.g., async logging backend).
//
// Ch11.6: Future extensions:
//   - Async log backend with dedicated writer thread
//   - Ring buffer for zero-allocation logging
//   - Structured logging (JSON output)

#include "task_scheduler/logger.hpp"

// Currently all Logger methods are inline in the header.
// This file reserves the translation unit for future non-template code.

# 📚 Documentation Center

> **Comprehensive documentation hub for the Raman Spectroscopy Analysis Application**

## 📖 Overview

This directory contains all project documentation, organized by component type for easy navigation and maintenance. All development work should reference and update these documents as appropriate.

### 🆕 Recent Major Updates (October 2025)
- **October 8 UI/UX Polish** ✨
  - See [`OCTOBER_8_2025_UI_IMPROVEMENTS.md`](./OCTOBER_8_2025_UI_IMPROVEMENTS.md) for complete details
  - Fixed critical bug: Pipeline steps disappearing when selecting multiple datasets
  - Enhanced confirmation dialog: Prominent output name, dataset checkboxes, output grouping options
  - Optimized layout proportions: Better parameter/visualization section balance
  - Simplified dialog header: 30% more compact while maintaining clarity
  - Cleaned debug logging: Production-ready, error-focused logging
- **Advanced Preprocessing Methods**: 6 new research-based preprocessing methods implemented for MGUS/MM classification
  - See [`reports_sumamry/PREPROCESSING_ENHANCEMENT_COMPLETE.md`](./reports_sumamry/PREPROCESSING_ENHANCEMENT_COMPLETE.md) for full details
  - New methods: Quantile Normalization, Rank Transform, PQN, Peak-Ratio Features, Butterworth High-Pass, CDAE
  - New category: Feature Engineering (dimensionality reduction)
- **Critical Bug Fixes**: Fixed Derivative parameter empty field, enumerate bug in feature engineering, deep learning syntax
- **UI Enhancements**: Pipeline step selection highlighting, compact layout optimizations

## 🗂️ Directory Structure

```
.docs/
├── README.md                    # This file - documentation guide
├── TODOS.md                     # Centralized task management (★ Start here for tasks)
├── COMMIT_MESSAGE.md            # Commit message guidelines
├── IMPLEMENTATION_SUMMARY.md    # Overall implementation notes
├── SPRINT_SUMMARY.md            # Sprint retrospectives
├── SPRINT_SUMMARY_UI_IMPROVEMENTS.md # UI improvements sprint
│
├── core/                        # Core application documentation
│   ├── main.md                  # Main application architecture
│   └── utils.md                 # Utility functions documentation
│
├── pages/                       # Page-specific documentation
│   ├── preprocess_page.md       # Preprocessing interface
│   ├── data_package_page.md     # Data import/management
│   └── home_page.md             # Project management page
│
├── components/                  # Reusable component documentation
│   └── (component docs)
│
├── widgets/                     # Widget system documentation
│   ├── parameter-widgets-fixes.md
│   ├── enhanced-parameter-widgets.md
│   ├── parameter-constraint-system.md
│   └── (other widget docs)
│
├── functions/                   # Function library documentation
│   ├── PARAMETER_CONSTRAINTS.md
│   └── README.md
│
├── reports_sumamry/             # Implementation reports and summaries
│   └── PREPROCESSING_ENHANCEMENT_COMPLETE.md  # ★ October 2025: 6 new preprocessing methods
│
└── testing/                     # Test documentation and results
    └── (test reports by feature)
```

## 🎯 Quick Navigation

### For Development Tasks
1. **Start Here**: [`TODOS.md`](./TODOS.md) - Check current tasks and priorities
2. **Architecture**: [`main.md`](./main.md) - Understand overall structure
3. **Component Docs**: Navigate to specific folders for implementation details

### For Testing
- All test documentation goes in [`testing/`](./testing/)
- Create subdirectories for specific features being tested
- Include terminal output, screenshots, and validation results

### For Documentation Updates
- Update relevant `.md` files in their specific folders
- Always update `TODOS.md` when completing tasks
- Keep `CHANGELOG.md` (root) in sync with major changes

## 📝 Documentation Standards

### File Naming
- Use lowercase with hyphens: `my-component.md`
- Match source file names when documenting specific files
- Use descriptive names for topic-based documentation

### Content Structure
Each documentation file should include:
1. **Title and Purpose**: Clear description of what's documented
2. **Architecture Overview**: How it fits in the system
3. **Key Features**: Main capabilities and functions
4. **Implementation Details**: Code structure and patterns
5. **Usage Examples**: How to use/integrate
6. **Recent Changes**: Update log (or link to CHANGELOG)

### Code Examples
- Use proper syntax highlighting
- Include context and explanations
- Show both successful and error cases
- Keep examples concise but complete

## 🔗 Integration with .AGI-BANKS

The `.AGI-BANKS` folder contains AI agent knowledge base files that **reference** this documentation:

```
.AGI-BANKS/                      .docs/
├── BASE_MEMORY.md              → References all .docs content
├── PROJECT_OVERVIEW.md         → Links to main.md, pages/
├── FILE_STRUCTURE.md           → Maintains high-level structure
├── IMPLEMENTATION_PATTERNS.md  → Links to widgets/, functions/
├── RECENT_CHANGES.md           → Syncs with TODOS.md
├── DEVELOPMENT_GUIDELINES.md   → References testing/
└── HISTORY_PROMPT.md           → Archives completed tasks
```

**Workflow**:
1. `.AGI-BANKS` provides high-level context and patterns
2. `.docs` contains detailed implementation documentation
3. `TODOS.md` tracks current and future work
4. Both systems stay synchronized

## 🚀 Development Workflow

### Starting a New Task
```
1. Check .docs/TODOS.md for task details
2. Review relevant documentation in .docs/
3. Reference .AGI-BANKS for patterns and context
4. Implement changes
5. Update both .docs/ and .AGI-BANKS
6. Mark task complete in TODOS.md
```

### Adding New Features
```
1. Document in TODOS.md as planned task
2. Create/update relevant .docs/ files
3. Implement feature
4. Create test documentation in testing/
5. Update .AGI-BANKS references
6. Update CHANGELOG.md
```

### Testing Protocol
```
1. Create test documentation in .docs/testing/
2. Run tests with terminal validation
3. Document results (pass/fail, issues found)
4. Clean up test artifacts
5. Update TODOS.md with findings
```

## 📋 Current Focus Areas

### Active Development
- **UI Improvements**: Dataset selection highlighting, export functionality
- **Documentation**: Organizing and centralizing all docs
- **Testing**: Robust validation infrastructure

### Documentation Priorities
1. Ensure all pages have comprehensive docs
2. Document all widgets and components
3. Create testing documentation framework
4. Keep TODOS.md current

## 🔍 Finding Documentation

### By Component Type
- **Pages**: See `pages/` folder
- **Widgets**: See `widgets/` folder
- **Functions**: See `functions/` folder
- **Tests**: See `testing/` folder

### By Feature
- **Preprocessing**: `pages/preprocess_page.md`
- **Data Import**: `pages/data_package_page.md`
- **Parameters**: `widgets/enhanced-parameter-widgets.md`
- **File Loading**: `functions/README.md`

### By Task
- **All Tasks**: `TODOS.md`
- **Recent Changes**: `.AGI-BANKS/RECENT_CHANGES.md`
- **Patterns**: `.AGI-BANKS/IMPLEMENTATION_PATTERNS.md`

## 🛠️ Maintenance

### Regular Updates
- **Daily**: Update TODOS.md for task progress
- **Per Feature**: Update relevant component docs
- **Per Sprint**: Review and reorganize as needed
- **Major Changes**: Update README.md and navigation

### Quality Checks
- [ ] All new features have documentation
- [ ] Code examples are tested and working
- [ ] Links between documents are valid
- [ ] TODOS.md reflects actual project status
- [ ] .AGI-BANKS and .docs stay synchronized

## 📞 Documentation Guidelines

### Writing Style
- Clear and concise
- Technical but accessible
- Include examples
- Explain "why" not just "what"
- Use diagrams when helpful

### Code Documentation
- Document complex algorithms
- Explain design decisions
- Note dependencies
- Include usage examples
- Document edge cases

### Update Frequency
- **Immediate**: Bug fixes, critical changes
- **Daily**: Task progress, small updates
- **Weekly**: Review and consolidation
- **Monthly**: Major reorganization if needed

---

**Last Updated**: October 1, 2025  
**Maintained By**: Development Team  
**Questions?**: Check TODOS.md or .AGI-BANKS/BASE_MEMORY.md

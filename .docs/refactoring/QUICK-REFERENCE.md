# Refactoring Quick Reference

**Quick access guide for the preprocess_page.py refactoring plan**

## 📖 Full Plan Location
`.docs/refactoring/preprocess-page-refactoring-plan.md` (6,000+ words)

## 🎯 Quick Start

### Before You Begin
1. Read the full plan (30-45 minutes)
2. Create Git branch: `refactor/preprocess-page`
3. Set up tests: `tests/pages/test_preprocess_page.py`
4. Take baseline snapshot (screenshots + test export)

### Phase Execution Order
```
Phase 1: Setup (1-2h)         → Prep infrastructure
Phase 2: UI Builders (3-4h)   → Extract UI creation
Phase 3: Data Manager (3-4h)  → Extract data ops
Phase 4: Pipeline Mgr (4-5h)  → Extract pipeline logic
Phase 5: Executor (4-5h)      → Extract execution
Phase 6: Preview Mgr (4-5h)   → Extract visualization
Phase 7: State Mgr (2-3h)     → Extract state
Phase 8: Optimize (2-3h)      → Clean utilities
Phase 9: Update Main (3-4h)   → Simplify coordinator
Phase 10: Testing (4-6h)      → Validate everything
```

## 📋 Documentation Template

### Function Docstring
```python
def method_name(self, param1: type, param2: type = default) -> return_type:
    """
    One-line description of what this does.
    
    Detailed description if needed (algorithm, flow).
    
    Args:
        param1 (type): Description
        param2 (type, optional): Description. Defaults to value.
    
    Returns:
        return_type: Description
        
    Raises:
        ExceptionType: When this happens
        
    Use in:
        - file.py: Class.method()
        - file.py: Class.method2()
        
    Example:
        >>> obj.method_name("test", 123)
        True
        
    Note:
        Important info (side effects, performance, constraints)
    """
    # Implementation
```

### Class Docstring
```python
class ClassName:
    """
    One-line description of class purpose.
    
    Detailed description (responsibility, design pattern).
    
    Attributes:
        attr1 (type): Description
        attr2 (type): Description
        
    Use in:
        - file.py: ParentClass (composition)
        - file.py: OtherClass.method()
        
    Example:
        >>> obj = ClassName()
        >>> obj.method()
        
    Note:
        Important constraints (threading, lifecycle)
    """
```

## 📁 Target Module Structure

```
pages/preprocess_page.py (300 lines)
└── PreprocessPage(QWidget)
    ├── __init__() → Create all managers
    ├── _setup_ui() → Use UIBuilder
    ├── _connect_signals() → Connect UI ↔ managers
    └── Public API (10-15 delegation methods)

pages/preprocess_page_utils/
├── ui_builders.py (400 lines)
│   └── UIBuilder
│       ├── build_pipeline_group()
│       ├── build_input_datasets_group()
│       ├── build_output_group()
│       └── build_visualization_group()
│
├── data_manager.py (400 lines)
│   └── DataManager
│       ├── load_project_data()
│       ├── export_dataset()
│       └── get_selected_datasets()
│
├── pipeline_manager.py (450 lines)
│   └── PipelineManager
│       ├── add_step()
│       ├── remove_step()
│       ├── load_from_json()
│       └── save_to_memory()
│
├── preprocessing_executor.py (400 lines)
│   ├── PreprocessingThread(QThread)
│   └── PreprocessingExecutor
│       ├── start_preprocessing()
│       └── handle_completion()
│
├── preview_manager.py (450 lines)
│   └── PreviewManager
│       ├── toggle_preview()
│       ├── update_preview()
│       └── manual_focus()
│
└── state_manager.py (250 lines)
    └── StateManager
        ├── save_to_memory()
        ├── restore_from_memory()
        └── show_history()
```

## ✅ Validation Checklist (Per Phase)

### After Each Phase
- [ ] All new files have docstrings
- [ ] All methods have structured comments
- [ ] Tests pass (no regressions)
- [ ] Code compiles without errors
- [ ] Git commit with descriptive message
- [ ] Tag stable point (e.g., `phase-2-complete`)

### Critical Validations
- [ ] **Phase 2**: UI renders identically to baseline
- [ ] **Phase 3**: Data load/export works correctly
- [ ] **Phase 4**: Pipeline add/remove/reorder works
- [ ] **Phase 5**: Preprocessing executes successfully
- [ ] **Phase 6**: Preview updates correctly
- [ ] **Phase 10**: All features work end-to-end

## 🚨 Common Pitfalls to Avoid

### ❌ Don't Do This
1. ❌ Extract multiple phases at once → Easy to break things
2. ❌ Skip validation checkpoints → Hidden bugs accumulate
3. ❌ Forget docstrings → Defeats maintainability goal
4. ❌ Change behavior during refactor → Scope creep
5. ❌ Test only at the end → Hard to find root cause

### ✅ Do This Instead
1. ✅ One phase at a time → Easier to debug
2. ✅ Validate after every change → Catch bugs early
3. ✅ Write docstrings first → Forces clear design
4. ✅ Preserve exact behavior → Refactor = restructure, not rewrite
5. ✅ Test continuously → Confidence in changes

## 🎯 Success Metrics (Final)

### File Size
- Main file: 3,068 → <300 lines (-90%)
- Max module: <500 lines each

### Code Quality
- Test coverage: >80%
- Cyclomatic complexity: <10 per method
- Documentation: 100% methods

### Functionality
- Zero regressions
- All features work
- Performance maintained

## 📞 When to Get Help

### Stop and Ask If:
1. Tests fail after extraction
2. Behavior changes unexpectedly
3. Circular dependencies appear
4. Performance degrades significantly
5. Unclear how to split a method

### Debug Strategy
1. Check git diff (what changed?)
2. Run specific test (which feature broke?)
3. Compare with baseline snapshot
4. Rollback to previous phase
5. Review full plan section

## 🔗 Related Documents

- **Full Plan**: `.docs/refactoring/preprocess-page-refactoring-plan.md`
- **Architecture**: `.docs/pages/preprocess_page.md`
- **Session Summary**: `.docs/summaries/2025-10-06-pipeline-fix-and-refactoring-plan.md`
- **TODOs**: `.docs/todos/TODOS.md`
- **Base Memory**: `.AGI-BANKS/BASE_MEMORY.md`

## ⏱️ Time Estimates

| Phase | Optimistic | Realistic | Pessimistic |
|-------|-----------|-----------|-------------|
| Phase 1 | 1h | 1.5h | 2h |
| Phase 2 | 3h | 3.5h | 4h |
| Phase 3 | 3h | 3.5h | 4h |
| Phase 4 | 4h | 4.5h | 5h |
| Phase 5 | 4h | 4.5h | 5h |
| Phase 6 | 4h | 4.5h | 5h |
| Phase 7 | 2h | 2.5h | 3h |
| Phase 8 | 2h | 2.5h | 3h |
| Phase 9 | 3h | 3.5h | 4h |
| Phase 10 | 4h | 5h | 6h |
| **Total** | **30h** | **35.5h** | **41h** |

## 🎓 Learning Resources

### Design Patterns Used
- **Builder Pattern**: UIBuilder creates complex UIs
- **Manager Pattern**: Each manager owns one concern
- **Observer Pattern**: Qt signals for loose coupling
- **Facade Pattern**: PreprocessPage coordinates managers

### Best Practices
- **Single Responsibility**: One class, one job
- **Dependency Injection**: Pass dependencies in constructor
- **Interface Segregation**: Small, focused interfaces
- **Documentation First**: Docstring before code

---

**Last Updated**: October 6, 2025  
**Plan Version**: 1.0.0  
**Estimated Completion**: 1-2 weeks (full-time) or 3-4 weeks (part-time)

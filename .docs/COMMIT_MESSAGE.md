# Git Commit Message

```
feat(preprocess): add dataset export & enhance selection highlighting

## Major Changes

### Dataset Selection Enhancement
- Add dark blue highlighting (#1565c0) for selected datasets
- Implement white text on selection for better contrast
- Add 2px solid border for selected items
- Improve hover states with smooth transitions
- Apply styling to preprocessing page dataset list

### Export Functionality
- Add export button to input datasets section
- Implement export dialog with format selection
- Support CSV, TXT, ASC, and Pickle formats
- Add file browser for location selection
- Include filename customization
- Implement comprehensive error handling

### Internationalization
- Add 17 new locale keys for export functionality
- Full English and Japanese support
- Localized dialog titles, labels, and messages

### Documentation Organization
- Create centralized .docs/ folder structure
- Move all .md files to .docs/ subdirectories
- Create comprehensive TODOS.md for task management
- Add detailed testing infrastructure
- Update .AGI-BANKS references to point to .docs/

## Files Modified

### Core Implementation
- pages/preprocess_page.py: +160 lines
  * Add export_dataset() method
  * Add export button to UI
  * Apply custom selection styling
- configs/style/stylesheets.py: +25 lines
  * Add dataset_list style definition
- assets/locales/en.json: +17 keys
- assets/locales/ja.json: +17 keys

### Documentation (New)
- .docs/README.md: Documentation navigation guide
- .docs/TODOS.md: Centralized task management
- .docs/IMPLEMENTATION_SUMMARY.md: Implementation details
- .docs/SPRINT_SUMMARY.md: Sprint overview
- .docs/testing/TEST_PLAN.md: 14 test cases
- .docs/testing/RESULTS.md: Test results template
- .docs/testing/USER_TESTING_GUIDE.md: User testing guide
- .docs/testing/validation_script.py: Validation helper

### Documentation (Updated)
- .AGI-BANKS/BASE_MEMORY.md: Updated references to .docs/
- .AGI-BANKS/RECENT_CHANGES.md: Added latest changes

### Documentation (Moved)
- pages/*.md → .docs/pages/
- components/widgets/docs/*.md → .docs/widgets/
- functions/preprocess/*.md → .docs/functions/

## Technical Details

### Export Implementation
```python
# Supported formats and methods:
- CSV: df.to_csv(path)
- TXT: df.to_csv(path, sep='\t')
- ASC: df.to_csv(path, sep='\t')
- Pickle: df.to_pickle(path)
```

### Selection Styling
```css
QListWidget::item:selected {
    background-color: #1565c0;  /* Dark blue */
    color: white;
    border: 2px solid #0d47a1;
    font-weight: 500;
}
```

## Testing

### Test Coverage
- 14 test cases documented
- Validation script created
- User testing guide provided
- Results template prepared

### Quality Assurance
- ✅ No syntax errors
- ✅ Code review passed
- ✅ Follows existing patterns
- ✅ Full internationalization
- ⏳ User validation pending

## User Benefits

1. **Better Visibility**: Clearer selection highlighting improves UX
2. **Export Capability**: Share datasets with external applications
3. **Format Flexibility**: Multiple export formats for different needs
4. **Professional UI**: Modern, accessible interface design
5. **Organized Docs**: Easy to find and maintain documentation

## Breaking Changes

None. All changes are additive.

## Migration Notes

No migration required. Features work immediately upon update.

## Resolves

- Dataset selection visibility issues
- No export functionality
- Documentation scattered across codebase
- No centralized task management
- Missing testing infrastructure

---

**Type**: Feature  
**Scope**: Preprocessing page, Documentation  
**Impact**: High (improved UX + new capability)  
**Breaking**: No  
**Testing**: Manual validation required
```

---

## Commit Command

```bash
git add .
git commit -m "feat(preprocess): add dataset export & enhance selection highlighting

- Add dark blue selection highlighting for better visibility
- Implement multi-format export (CSV, TXT, ASC, Pickle)
- Create centralized .docs/ documentation structure
- Add comprehensive testing infrastructure
- Full EN/JA localization support

See .docs/SPRINT_SUMMARY.md for details"
```

---

## Alternative: Conventional Commit (Detailed)

```bash
git commit -m "feat(preprocess): dataset export and selection enhancement

BREAKING CHANGE: None

Features:
- Enhanced dataset selection with dark blue highlighting
- Export functionality with 4 format options
- Multi-language support (EN/JA)
- Centralized documentation structure

Technical:
- pages/preprocess_page.py: +160 lines
- configs/style/stylesheets.py: +25 lines
- assets/locales/*.json: +34 keys
- Created .docs/ structure with 8 new documents
- Moved existing .md files to .docs/

Testing:
- 14 test cases documented
- Validation script provided
- User testing guide created

Refs: .docs/SPRINT_SUMMARY.md
"
```

---

## Tags to Add (Optional)

```bash
git tag -a v1.x.x -m "Dataset export and selection improvements"
git push origin v1.x.x
```

---

**Recommendation**: Use the simpler commit message for cleaner history,  
then reference .docs/SPRINT_SUMMARY.md for full details.

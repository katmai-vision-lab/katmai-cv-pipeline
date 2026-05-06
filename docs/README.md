# Documentation Index

This directory contains comprehensive documentation for the Katmai CV Pipeline project.

## 📚 Quick Navigation

### 🐻 Bear Detection System
- **[Bear Auto-Annotation System Report](bear_auto_annotation_system_report.md)** - Complete technical report on multi-model consensus bear detection

### 🐟 Salmon Detection System  
- **[Salmon Detection Guide](SALMON_DETECTION_GUIDE.md)** ⭐ - Complete user guide with detailed workflows
- **[Salmon Quick Reference](SALMON_QUICK_REFERENCE.md)** ⚡ - One-page command reference
- **[Salmon Auto-Annotation System Report](salmon_auto_annotation_system_report.md)** - Technical report on Stacking meta-learner

### 📋 Team Resources
- **[Team Scrum Best Practices](team-scrum-best-practice.md)** - Team collaboration guidelines

---

## 🎯 Documentation by Use Case

### I want to... detect salmon in my videos
1. Start with: **[Salmon Quick Reference](SALMON_QUICK_REFERENCE.md)**
2. If you need details: **[Salmon Detection Guide](SALMON_DETECTION_GUIDE.md)**

### I want to... understand how the salmon system works
- Read: **[Salmon Auto-Annotation System Report](salmon_auto_annotation_system_report.md)**

### I want to... train a custom salmon detection model
- Follow: **[Salmon Detection Guide → Training Custom Stacking Model](SALMON_DETECTION_GUIDE.md#training-custom-stacking-model)**

### I want to... detect bears in videos
- Read: **[Bear Auto-Annotation System Report](bear_auto_annotation_system_report.md)**

### I want to... understand the technical differences
- Compare: **[Salmon Report §6](salmon_auto_annotation_system_report.md#6-comparison-with-the-bear-system)** vs **[Bear Report](bear_auto_annotation_system_report.md)**

---

## 📊 System Comparison

| Feature | Bear System | Salmon System |
|---------|-------------|---------------|
| **Method** | Weighted Voting | **Stacking Meta-Learner** |
| **Base Models** | GDINO + DETR + MegaDet | GDINO + OWL-ViT + Florence-2 |
| **Precision** | 89.3% | **97.5%** ⭐ |
| **Recall** | 99.8% ⭐ | 96.6% |
| **Manual Review** | Required | **None** ⭐ |
| **Calibration** | Isotonic Regression | Learned RF Classifier |
| **Feature Engineering** | None | **11 features** |
| **Documentation** | [Report](bear_auto_annotation_system_report.md) | [Report](salmon_auto_annotation_system_report.md) + [Guide](SALMON_DETECTION_GUIDE.md) |

---

## 📖 Document Descriptions

### Technical Reports

**[salmon_auto_annotation_system_report.md](salmon_auto_annotation_system_report.md)** (70 pages)
- Comprehensive technical report on Stacking meta-learner approach
- Includes methodology, experiments, ablation studies, feature importance
- Performance: 97.5% precision, 96.6% recall, 99.9% AUC-ROC
- Comparison with traditional voting methods
- Section highlights:
  - §2: Methodology (base models, prompt optimization, feature engineering)
  - §3: Experimental results (validation performance, feature importance)
  - §4: System implementation (code architecture, key functions)
  - §5: Usage guide (quick start, custom training)
  - §6: Comparison with bear system

**[bear_auto_annotation_system_report.md](bear_auto_annotation_system_report.md)** (525 lines)
- Technical report on multi-model consensus bear detection
- Weighted voting with probability calibration
- Performance: 89.3% precision, 99.8% recall on 341 test images

### User Guides

**[SALMON_DETECTION_GUIDE.md](SALMON_DETECTION_GUIDE.md)** (60 pages)
- Complete user guide for salmon detection system
- Target audience: Researchers, data scientists, pipeline users
- Sections:
  1. **Quick Start** - Get running in 5 minutes
  2. **Method Comparison** - Stacking vs Voting
  3. **Detailed Workflows** - Batch processing, custom training, comparison
  4. **Understanding Models** - Model characteristics, trustworthiness patterns
  5. **Training Guide** - When/how to train custom models
  6. **Troubleshooting** - Common issues and solutions
  7. **Performance Optimization** - Speed and memory tips
  8. **FAQ** - 20+ common questions

**[SALMON_QUICK_REFERENCE.md](SALMON_QUICK_REFERENCE.md)** (2 pages)
- One-page quick reference card
- Essential commands, parameters, troubleshooting
- Print-friendly format
- Perfect for: Quick lookup, sharing with team, cheat sheet

### Team Documentation

**[team-scrum-best-practice.md](team-scrum-best-practice.md)**
- Team collaboration guidelines
- Scrum practices and workflows

---

## 🚀 Getting Started Paths

### Path 1: New User (Just Want Results)
```
1. Read: Salmon Quick Reference (5 min)
2. Run: Commands from quick reference (10 min)
3. View: Visualized results
✓ You're done! You have salmon detections.
```

### Path 2: Power User (Understanding System)
```
1. Read: Salmon Detection Guide (30 min)
2. Read: Salmon Technical Report - §1-3 (20 min)
3. Experiment: Try different parameters
4. Compare: Stacking vs Voting on your data
✓ You understand the system deeply.
```

### Path 3: Researcher (Custom Training)
```
1. Read: Salmon Detection Guide - Training section (15 min)
2. Follow: Training workflow step-by-step (2-4 hours)
3. Read: Salmon Technical Report - §2.4-2.5 (Feature engineering)
4. Train: Custom model on your data
✓ You have a custom-trained model for your domain.
```

### Path 4: Technical Deep Dive
```
1. Read: Salmon Technical Report - Complete (1 hour)
2. Read: Bear Technical Report - Complete (1 hour)
3. Compare: §6 in Salmon Report (differences)
4. Explore: Code in src/preprocessing/annotation_salmon/
✓ You understand the entire technical architecture.
```

---

## 📁 Directory Structure

```
docs/
├── README.md (this file)                          # Documentation index
├── SALMON_DETECTION_GUIDE.md                      # Complete user guide
├── SALMON_QUICK_REFERENCE.md                      # One-page reference
├── salmon_auto_annotation_system_report.md        # Technical report
├── bear_auto_annotation_system_report.md          # Bear system report
├── team-scrum-best-practice.md                    # Team guidelines
├── design-docs/                                   # Design documents
├── images/                                        # Documentation images
├── meeting-summaries/                             # Meeting notes
├── templates/                                     # Document templates
└── workflow-docs/                                 # Workflow documentation
```

---

## 🔗 External Resources

- **Project Repository**: https://github.com/katmai-vision-lab
- **SharePoint**: [UW Katmai Vision Lab](https://uwnetid.sharepoint.com/sites/katmai-vision-lab)
- **Main README**: [../README.md](../README.md)

---

## 📝 Contributing to Documentation

### Adding New Documentation

1. **Create document** in appropriate location
2. **Add entry** to this README
3. **Update navigation** sections
4. **Test all links**
5. **Submit PR**

### Documentation Standards

- **Format**: Markdown (.md)
- **Structure**: Clear hierarchy (H1 → H2 → H3)
- **Code blocks**: Always specify language
- **Links**: Use relative paths
- **Images**: Store in `images/` directory
- **Examples**: Include working code examples
- **Updates**: Date stamp major revisions

### Documentation Types

| Type | When to Create | Template | Examples |
|------|----------------|----------|----------|
| **Technical Report** | System implementation complete | See existing reports | salmon_auto_annotation_system_report.md |
| **User Guide** | Complex workflow needs explanation | See SALMON_DETECTION_GUIDE.md | SALMON_DETECTION_GUIDE.md |
| **Quick Reference** | Users need fast lookup | See SALMON_QUICK_REFERENCE.md | SALMON_QUICK_REFERENCE.md |
| **Meeting Notes** | After important meetings | templates/ | meeting-summaries/ |
| **Design Doc** | Before implementing major feature | templates/ | design-docs/ |

---

## 🆘 Help & Support

### Documentation Issues
- **Missing info**: Open GitHub issue with "docs" label
- **Unclear instructions**: Comment on specific section
- **Broken links**: Report via GitHub issue
- **Suggestions**: Open discussion thread

### Getting Help
1. **Check documentation** (you're here!)
2. **Search GitHub issues**
3. **Ask on SharePoint**
4. **Open new GitHub issue**

---

## 📊 Documentation Metrics

### Salmon System Documentation
- **Technical Report**: 70 pages, 8 sections, 15 figures/tables
- **User Guide**: 60 pages, 9 sections, 20+ FAQ items
- **Quick Reference**: 2 pages, essential commands only
- **Code Comments**: 415 lines in train_stacking.py, 320+ in predict_stacking.py
- **Total**: ~135 pages of documentation

### Coverage
- ✅ Installation and setup
- ✅ Quick start (5 min)
- ✅ Detailed workflows
- ✅ Technical methodology
- ✅ Performance metrics
- ✅ Troubleshooting
- ✅ FAQ
- ✅ API reference
- ✅ Comparison with alternatives
- ✅ Future roadmap

---

**Last Updated**: March 4, 2026  
**Documentation Version**: 1.0  
**Maintained by**: Katmai Vision Lab

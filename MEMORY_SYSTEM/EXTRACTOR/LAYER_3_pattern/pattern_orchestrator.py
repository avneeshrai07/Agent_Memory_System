from MEMORY_SYSTEM.EXTRACTOR.LAYER_3_pattern.pattern_detector import PatternDetector

async def run_layer3_patterns(user_id: str):
    detector = PatternDetector()
    
    patterns = {
        'preferences': await detector.detect_preferences(user_id),
        'domains': await detector.detect_domains(user_id),
        'constraints': await detector.detect_constraints(user_id)
    }
    
    # Store in patterns table (create if needed)
    return patterns
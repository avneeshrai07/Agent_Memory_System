# MEMORY_SYSTEM/EXTRACTOR/LAYER_3_pattern/context_builder.py

from typing import Dict, Any, List
import json


def build_smart_context(user_patterns: Dict[str, Any]) -> str:
    """
    Extract the most relevant insights from user patterns.
    Returns a concise, actionable context string for the LLM.
    
    Priority:
    1. Persona (tone/style)
    2. Top 3 preferences (by frequency)
    3. Primary domain (highest confidence)
    4. Critical constraints
    """
    
    context_parts = []
    
    # ========================================================================
    # 1. PERSONA: Extract tone, style, template
    # ========================================================================
    if user_patterns.get('persona'):
        persona = user_patterns['persona']
        metadata = json.loads(persona['metadata']) if isinstance(persona['metadata'], str) else persona['metadata']
        
        tone = metadata.get('tone', 'neutral')
        style = metadata.get('style', 'balanced')
        template = metadata.get('template_type', 'general')
        
        context_parts.append(f"## User Communication Profile")
        context_parts.append(f"- **Tone**: {tone.upper()}")
        context_parts.append(f"- **Style**: {style.title()} ({'concise and direct' if style == 'concise' else 'comprehensive' if style == 'detailed' else 'balanced'})")
        context_parts.append(f"- **Domain**: {template.replace('_', ' ').title()}")
        context_parts.append("")
    
    # ========================================================================
    # 2. TOP PREFERENCES: Select top 3 by frequency
    # ========================================================================
    preferences = user_patterns.get('preferences', [])
    if preferences:
        # Sort by frequency (descending)
        top_prefs = sorted(preferences, key=lambda x: x['frequency'], reverse=True)[:3]
        
        context_parts.append(f"## User Preferences")
        for pref in top_prefs:
            freq = pref['frequency']
            name = pref['pattern_name'].replace('_', ' ').title()
            confidence = pref['confidence']
            context_parts.append(f"- **{name}** (mentioned {freq}x, confidence: {confidence:.0%})")
        context_parts.append("")
    
    # ========================================================================
    # 3. PRIMARY DOMAIN: Highest confidence domain
    # ========================================================================
    domains = user_patterns.get('domains', [])
    if domains:
        # Sort by confidence (descending)
        primary_domain = sorted(domains, key=lambda x: x['confidence'], reverse=True)[0]
        
        metadata = json.loads(primary_domain['metadata']) if isinstance(primary_domain['metadata'], str) else primary_domain['metadata']
        share = metadata.get('share_of_conversations', 'N/A')
        
        context_parts.append(f"## Primary Work Domain")
        context_parts.append(f"- **{primary_domain['pattern_name'].replace('_', ' ').title()}** ({share} of conversations)")
        context_parts.append(f"- {primary_domain['description']}")
        context_parts.append("")
    
    # ========================================================================
    # 4. CONSTRAINTS: All constraints (high importance)
    # ========================================================================
    constraints = user_patterns.get('constraints', [])
    if constraints:
        context_parts.append(f"## Important Constraints")
        for constraint in constraints:
            metadata = json.loads(constraint['metadata']) if isinstance(constraint['metadata'], str) else constraint['metadata']
            fact = metadata.get('constraint_fact', constraint['description'])
            context_parts.append(f"- ⚠️ {fact}")
        context_parts.append("")
    
    # ========================================================================
    # 5. EXPERTISE: Show if exists
    # ========================================================================
    expertise = user_patterns.get('expertise', [])
    if expertise:
        context_parts.append(f"## User Expertise")
        for exp in expertise[:3]:  # Top 3
            context_parts.append(f"- {exp['pattern_name'].replace('_', ' ').title()}")
        context_parts.append("")
    
    return "\n".join(context_parts)


def build_compact_context(user_patterns: Dict[str, Any]) -> str:
    """
    Ultra-compact version for token-limited scenarios.
    Returns only the absolute essentials.
    """
    parts = []
    
    # Persona
    if user_patterns.get('persona'):
        persona = user_patterns['persona']
        metadata = json.loads(persona['metadata']) if isinstance(persona['metadata'], str) else persona['metadata']
        parts.append(f"Tone: {metadata.get('tone', 'neutral')}, Style: {metadata.get('style', 'balanced')}")
    
    # Top preference
    preferences = user_patterns.get('preferences', [])
    if preferences:
        top_pref = max(preferences, key=lambda x: x['frequency'])
        parts.append(f"Prefers: {top_pref['pattern_name'].replace('_', ' ')}")
    
    # Primary domain
    domains = user_patterns.get('domains', [])
    if domains:
        top_domain = max(domains, key=lambda x: x['confidence'])
        parts.append(f"Works with: {top_domain['pattern_name'].replace('_', ' ')}")
    
    # Constraints
    constraints = user_patterns.get('constraints', [])
    if constraints:
        parts.append(f"Constraint: {constraints[0]['pattern_name']}")
    
    return " | ".join(parts)

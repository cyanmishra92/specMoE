#!/usr/bin/env python3
"""
Data Requirements Estimator for Switch Transformer Fine-tuning
"""

def estimate_data_requirements():
    """Estimate data requirements for fine-tuning"""
    
    print("📊 DATA REQUIREMENTS ESTIMATION")
    print("=" * 50)
    
    # Typical research paper statistics
    avg_pages_per_paper = 8  # Research papers: 6-12 pages
    words_per_page = 500     # Academic text density
    tokens_per_word = 1.3    # GPT-2 tokenizer average
    
    num_papers = 12  # Your target
    
    total_pages = num_papers * avg_pages_per_paper
    total_words = total_pages * words_per_page
    total_tokens = int(total_words * tokens_per_word)
    
    # Sequence parameters
    seq_length = 512
    overlap = 0.1  # 10% overlap between sequences
    effective_length = int(seq_length * (1 - overlap))
    
    estimated_sequences = total_tokens // effective_length
    
    print(f"📄 Input: {num_papers} research papers")
    print(f"📃 Estimated pages: {total_pages}")
    print(f"📝 Estimated words: {total_words:,}")
    print(f"🔤 Estimated tokens: {total_tokens:,}")
    print(f"📦 Training sequences: {estimated_sequences:,}")
    print()
    
    # Fine-tuning requirements
    print("🎯 FINE-TUNING ASSESSMENT")
    print("=" * 50)
    
    min_sequences_finetuning = 1000   # Minimum for meaningful fine-tuning
    ideal_sequences_finetuning = 5000  # Ideal for good adaptation
    
    print(f"✅ Minimum sequences needed: {min_sequences_finetuning:,}")
    print(f"🎯 Ideal sequences: {ideal_sequences_finetuning:,}")
    print(f"📊 Your estimated sequences: {estimated_sequences:,}")
    print()
    
    if estimated_sequences >= ideal_sequences_finetuning:
        print("🟢 EXCELLENT: You have ideal data for fine-tuning!")
        quality = "Excellent"
    elif estimated_sequences >= min_sequences_finetuning:
        print("🟡 GOOD: Sufficient data for fine-tuning")
        quality = "Good"
    else:
        print("🟠 LIMITED: May need more papers or longer papers")
        quality = "Limited"
        papers_needed = (min_sequences_finetuning * effective_length) // (avg_pages_per_paper * words_per_page * tokens_per_word)
        print(f"   Recommend: {papers_needed} papers minimum")
    
    print()
    
    # Training splits
    train_sequences = int(estimated_sequences * 0.8)
    val_sequences = int(estimated_sequences * 0.1)
    test_sequences = estimated_sequences - train_sequences - val_sequences
    
    print("📊 TRAINING SPLITS")
    print("=" * 50)
    print(f"🏋️ Training: {train_sequences:,} sequences (80%)")
    print(f"🔍 Validation: {val_sequences:,} sequences (10%)")
    print(f"🧪 Test: {test_sequences:,} sequences (10%)")
    print()
    
    # Time and resource estimates
    print("⏱️ TRAINING ESTIMATES (RTX 3090)")
    print("=" * 50)
    
    # Fine-tuning estimates
    batch_size = 4  # Conservative for 24GB VRAM
    steps_per_epoch = train_sequences // batch_size
    epochs = 3  # Typical for fine-tuning
    total_steps = steps_per_epoch * epochs
    seconds_per_step = 2.5  # Estimated for Switch model
    
    total_time_hours = (total_steps * seconds_per_step) / 3600
    
    print(f"📦 Batch size: {batch_size}")
    print(f"🔄 Steps per epoch: {steps_per_epoch:,}")
    print(f"🔃 Epochs: {epochs}")
    print(f"📈 Total training steps: {total_steps:,}")
    print(f"⏱️ Estimated training time: {total_time_hours:.1f} hours")
    print()
    
    # Memory estimates
    print("💾 MEMORY ESTIMATES")
    print("=" * 50)
    print(f"🧠 Model size: ~2-3 GB")
    print(f"📊 Training state: ~8-12 GB")
    print(f"🔄 Gradients: ~3-4 GB")
    print(f"📦 Batch data: ~1-2 GB")
    print(f"🎯 Total VRAM needed: ~14-21 GB")
    print(f"✅ RTX 3090 VRAM: 24 GB (Perfect fit!)")
    print()
    
    return {
        'papers': num_papers,
        'estimated_sequences': estimated_sequences,
        'quality': quality,
        'training_time_hours': total_time_hours,
        'vram_needed_gb': '14-21',
        'feasible': True
    }

if __name__ == "__main__":
    estimate_data_requirements()
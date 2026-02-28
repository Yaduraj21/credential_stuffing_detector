def extract_features(raw_login_data):
    """
    Converts raw backend data into a numerical array for the ML model.
    """
    # 1. Calculate Failed Login Ratio
    total = raw_login_data['total_attempts']
    ratio = raw_login_data['failed_attempts'] / total if total > 0 else 0
    
    # 2. Boolean flags (0 or 1)
    device_flag = 1 if raw_login_data['is_new_device'] else 0
    geo_flag = 1 if raw_login_data['is_impossible_travel'] else 0
    honeypot_flag = 1 if raw_login_data['honeypot_filled'] else 0
    
    # 3. Combine into the Feature Vector
    return [
        raw_login_data['attempts_per_min'],
        ratio,
        raw_login_data['unique_accounts_count'],
        device_flag,
        geo_flag,
        honeypot_flag
    ]
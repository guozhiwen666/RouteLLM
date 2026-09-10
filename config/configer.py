class Configer:
    #工程参数
    ROUTELLM_THRESHOLD_MAX_STEP=1       #最大步长阈值
    ROUTELLM_THRESHOLD_COOLDOWN_S=2     #阈值冷却时间（秒）
    ROUTELLM_ROUTE_BUDGET_MS=3          #路由预算毫秒数
    ROUTELLM_SELF_EVAL_TIMEOUT_S=4      #自我评估超时时间（秒）
    ROUTELLM_SLM_TIMEOUT_S=5            #slm 超时秒数
    ROUTELLM_CLOUD_TIMEOUT_S=6          #云超时秒数
    ROUTELLM_JUDGE_TIMEOUT_S=7          #判定超时秒数
    ROUTELLM_MAX_RETRIES=1              #最大重试次数
    ROUTELLM_BACKOFF_BASE_S=8           #退避基数秒
    ROUTELLM_MIN_OUTPUT_TOKENS=1        #最小输出词元数
    ROUTELLM_MAX_REPEAT_RATIO=1         #最大语法重复比例
    ROUTELLM_WINDOW_MIN_SAMPLES=1       #窗口最小样本数
    ROUTELLM_CACHE_LEN_RATIO_MIN=1      #最小缓存长度比率
    ROUTELLM_CACHE_LEN_RATIO_MAX=1      #最大缓存长度比率

configer=Configer()
def currency_fmt(v, symbol=''):
    try:
        if v is None:
            return 'N/A'
        return f"{symbol}{v:,.2f}"
    except Exception:
        return str(v)

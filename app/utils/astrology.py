from datetime import date


def get_sun_sign(date_of_birth: date) -> str:
    month_day = (date_of_birth.month, date_of_birth.day)
    signs = (
        ((3, 21), (4, 19), "Aries"),
        ((4, 20), (5, 20), "Taurus"),
        ((5, 21), (6, 20), "Gemini"),
        ((6, 21), (7, 22), "Cancer"),
        ((7, 23), (8, 22), "Leo"),
        ((8, 23), (9, 22), "Virgo"),
        ((9, 23), (10, 22), "Libra"),
        ((10, 23), (11, 21), "Scorpio"),
        ((11, 22), (12, 21), "Sagittarius"),
        ((12, 22), (1, 19), "Capricorn"),
        ((1, 20), (2, 18), "Aquarius"),
        ((2, 19), (3, 20), "Pisces"),
    )
    for start, end, sign in signs:
        if start <= end and start <= month_day <= end:
            return sign
        if start > end and (month_day >= start or month_day <= end):
            return sign
    raise ValueError(f"Unsupported birth date: {date_of_birth}")

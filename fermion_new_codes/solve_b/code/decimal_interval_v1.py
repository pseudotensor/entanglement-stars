"""Outward Decimal intervals: directed basic operations, bracketed sqrt/ln.

API contract: https://docs.python.org/3/library/decimal.html
Decimal basic operations honor context rounding; ln is correctly rounded
half-even. sqrt endpoints are additionally checked as exact rationals.
"""
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
from fractions import Fraction
import math

PRECISION=80
DOWN=Context(prec=PRECISION,rounding=ROUND_FLOOR,Emin=-999999,Emax=999999)
UP=Context(prec=PRECISION,rounding=ROUND_CEILING,Emin=-999999,Emax=999999)
NEAR=Context(prec=PRECISION,rounding=ROUND_HALF_EVEN,Emin=-999999,Emax=999999)


class Interval:
    def __init__(self,value=0,upper=None):
        if isinstance(value,Interval):self.lo,self.hi=value.lo,value.hi;return
        if upper is not None:self.lo,self.hi=Decimal(value),Decimal(upper)
        elif isinstance(value,float):
            if not math.isfinite(value):raise ValueError('nonfinite binary64 input')
            numerator,denominator=value.as_integer_ratio()
            self.lo=DOWN.divide(Decimal(numerator),Decimal(denominator))
            self.hi=UP.divide(Decimal(numerator),Decimal(denominator))
        elif isinstance(value,Fraction):
            self.lo=DOWN.divide(Decimal(value.numerator),Decimal(value.denominator))
            self.hi=UP.divide(Decimal(value.numerator),Decimal(value.denominator))
        else:self.lo=self.hi=Decimal(value)
        if not self.lo.is_finite() or not self.hi.is_finite() or self.lo>self.hi:raise ValueError('invalid interval')

    def __add__(self,other):
        other=Interval(other);return Interval(DOWN.add(self.lo,other.lo),UP.add(self.hi,other.hi))
    __radd__=__add__
    def __neg__(self):return Interval(self.hi.copy_negate(),self.lo.copy_negate())
    def __sub__(self,other):return self+-Interval(other)
    def __rsub__(self,other):return Interval(other)+-self
    def __mul__(self,other):
        other=Interval(other);pairs=[(a,b) for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        return Interval(min(DOWN.multiply(a,b) for a,b in pairs),max(UP.multiply(a,b) for a,b in pairs))
    __rmul__=__mul__
    def __truediv__(self,other):
        other=Interval(other)
        if other.lo<=0<=other.hi:raise ValueError('interval denominator includes zero')
        return self*Interval(DOWN.divide(Decimal(1),other.hi),UP.divide(Decimal(1),other.lo))
    def __rtruediv__(self,other):return Interval(other)/self
    def __abs__(self):
        if self.lo>=0:return Interval(self)
        if self.hi<=0:return -self
        return Interval(0,max(self.lo.copy_abs(),self.hi.copy_abs()))
    def sqrt(self):
        if self.lo<0:raise ValueError('negative square-root interval')
        def endpoint(value,lower):
            root=NEAR.sqrt(value)
            # Exact rational comparisons certify the chosen enclosing endpoint,
            # independently of which nearest root the Decimal backend returns.
            target=Fraction(value)
            if lower:
                while Fraction(root)**2>target:root=NEAR.next_minus(root)
            else:
                while Fraction(root)**2<target:root=NEAR.next_plus(root)
            return root
        return Interval(endpoint(self.lo,True),endpoint(self.hi,False))
    def ln(self):
        if self.lo<=0:raise ValueError('nonpositive logarithm interval')
        def endpoint(value,lower):
            if value==1:return Decimal(0)
            rounded=NEAR.ln(value)
            return NEAR.next_minus(rounded) if lower else NEAR.next_plus(rounded)
        return Interval(endpoint(self.lo,True),endpoint(self.hi,False))
    def cap(self,value):
        value=Decimal(value);return Interval(min(self.lo,value),min(self.hi,value))
    def as_dict(self):return {'lower':str(self.lo),'upper':str(self.hi)}
    def contains_fraction(self,value):return Fraction(self.lo)<=value<=Fraction(self.hi)


def positive(value,label):
    value=Interval(value)
    if value.lo<=0:raise ValueError(label+' not strictly positive')
    return value


def binary_entropy_increasing(value):
    """Monotone binary entropy on [0,1/2], not an entropy-trace evaluation."""
    value=Interval(value)
    if value.lo<0 or value.hi>Decimal('.5'):raise ValueError('entropy modulus argument outside monotone interval')
    def at(x):
        if x==0:return Interval(0)
        if x==Decimal('.5'):return Interval(2).ln()
        point=Interval(x)
        return -point*point.ln()-(1-point)*(1-point).ln()
    return Interval(at(value.lo).lo,at(value.hi).hi)

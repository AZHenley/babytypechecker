from typing import Union, Optional

a: Union[int, str] = 1  
b: Union[bool, str] = "Hi"
c: Optional[int] = None
d: Union[bool, int] = "oops"   # ❌ str not in union
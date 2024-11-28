from dash import html
from enum import Enum


class TypographyVariants(Enum):
    HEADING1 = "heading1"
    HEADING2 = "heading2"
    HEADING3 = "heading3"
    BODY1 = ("BODY1",)
    BODY2 = "BODY2"


def getTypographyStyles(typgraphy: TypographyVariants, extraClassName):
    className = ""
    if typgraphy == "heading1":
        className += "text-[42px] font-semibold "
    if typgraphy == "heading2":
        className += "text-[36px] font-semibold "
    if typgraphy == "heading3":
        className += "text-[24px] font-semibold "
    if typgraphy == "body1":
        className += "text-[20px] font-medium "
    if typgraphy == "body2":
        className += "text-[16] font-medium "
    return className + extraClassName


def P(children: any, variant: TypographyVariants, className="", **args):
    utilizedClassName = getTypographyStyles(variant, className)
    return html.P(children=children, className=utilizedClassName, **args)

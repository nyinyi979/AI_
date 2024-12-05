from dash import html
from enum import Enum


class ButtonVariants(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    WARNING = "warning"
    SUCCESS = "success"
    ERROR = "error"
    PRIMARYGHOST = "primary_ghost"
    SECONDARYGHOST = "secondary_ghost"
    WARNINGHOST = "warning_ghost"
    SUCCESSGHOST = "success_ghost"
    ERRORGHOST = "error_ghost"


class ButtonSizes(Enum):
    SMALL = "sm"
    MEDIUM = "md"


def getButtonStyles(extraClassName: str, variant: ButtonVariants, size=ButtonSizes):
    className = "duration-300 disabled-[#424242] font-semibold "
    if variant == "primary":
        className += "bg-[#C4DFDF] hover:bg-[#B1CBCB] group-hover:bg-[#B1CBCB] disabled:bg-[#9BADAD]"
    elif variant == "secondary":
        className += "bg-[#FFEEEE] hover:bg-[#F9EAEA] group-hover:bg-[#F9EAEA] disabled:bg-[#E8D7D7]"
    elif variant == "success":
        className += "bg-[#A2E995] hover:bg-[#8AC67F] group-hover:bg-[#8AC67F] disabled:bg-[#6D9865]"
    elif variant == "warning":
        className += "bg-[#FADFA1] hover:bg-[#DFC790] group-hover:bg-[#DFC790] disabled:bg-[#C1AC7C]"
    elif variant == "error":
        className += "bg-[#E55757] hover:bg-[#CA4B4B] group-hover:bg-[#CA4B4B] disabled:bg-[#A93A3A]"
    elif variant == "primary_ghost":
        className += "border border-[#C4DFDF] hover:bg-[#E3F4F4] group-hover:bg-[#E3F4F4] disabled:border-[#9BADAD]"
    elif variant == "secondary_ghost":
        className += "border border-[#FFEEEE] hover:bg-[#FFF6F6] group-hover:bg-[#FFF6F6] disabled:border-[#E8D7D7]"
    elif variant == "success_ghost":
        className += "border border-[#A2E995] hover:bg-[#B3FFA5] group-hover:bg-[#B3FFA5] disabled:border-[#6D9865]"
    elif variant == "warning_ghost":
        className += "border border-[#FADFA1] hover:bg-[#FCE1A5] group-hover:bg-[#FCE1A5] disabled:border-[#C1AC7C]"
    elif variant == "error_ghost":
        className += "border border-[#E55757] hover:bg-[#FF6565] group-hover:bg-[#FF6565] disabled:border-[#A93A3A]"

    if size == "sm":
        className += " p-2.5 rounded-lg text-base "
    else:
        className += " py-4 px-5 rounded-xl text-lg "
    return className + extraClassName


def Button(
    children: any,
    className="",
    asLink=False,
    variant=ButtonVariants,
    size=ButtonSizes,
    **args
):
    utilizedClassName = getButtonStyles(className, variant, size)
    return (
        html.A(children=children, className=utilizedClassName, **args)
        if asLink
        else html.Button(children=children, className=utilizedClassName, **args)
    )

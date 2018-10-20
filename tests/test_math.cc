#include <catch.hpp>

#include "fn.h"

TEST_CASE("Addition and subtraction")
{
    REQUIRE(add(1, 1) == 2);
    REQUIRE(subtract(1, 1) == 0);
}

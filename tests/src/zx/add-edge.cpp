#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <utility>

#include "util/phase.hpp"
#include "zx/zx_def.hpp"
#include "zx/zxgraph.hpp"

using namespace qsyn::zx;
using dvlab::Phase;

TEST_CASE("Adding Edges between Z-Z/X-X connections", "[zx][basic]") {
    ZXGraph g;

    auto [vt1, vt2] = GENERATE(std::make_pair(VertexType::z, VertexType::z),
                               std::make_pair(VertexType::x, VertexType::x));
    g.add_vertex(vt1);
    g.add_vertex(vt2);

    SECTION("Add simple edge") {
        g.add_edge(g[0], g[1], EdgeType::simple);
        REQUIRE(g.get_edge_type(g[0], g[1]) == EdgeType::simple);

        SECTION("Add another simple edge") {
            g.add_edge(g[0], g[1], EdgeType::simple);
            REQUIRE(g.get_edge_type(g[0], g[1]) == EdgeType::simple);
            REQUIRE(g[0]->phase() == Phase(0));
            REQUIRE(g[1]->phase() == Phase(0));
        }

        SECTION("Add another hadamard edge") {
            g.add_edge(g[0], g[1], EdgeType::hadamard);
            REQUIRE(g.get_edge_type(g[0], g[1]) == EdgeType::simple);
            REQUIRE(g[0]->phase() == Phase(1));
            REQUIRE(g[1]->phase() == Phase(0));
        }
    }

    SECTION("Add hadamard edge") {
        g.add_edge(g[0], g[1], EdgeType::hadamard);
        REQUIRE(g.get_edge_type(g[0], g[1]) == EdgeType::hadamard);

        SECTION("Add another simple edge") {
            g.add_edge(g[0], g[1], EdgeType::simple);
            REQUIRE(g.get_edge_type(g[0], g[1]) == EdgeType::simple);
            REQUIRE(g[0]->phase() == Phase(1));
            REQUIRE(g[1]->phase() == Phase(0));
        }

        SECTION("Add another hadamard edge") {
            g.add_edge(g[0], g[1], EdgeType::hadamard);
            REQUIRE(g.get_edge_type(g[0], g[1]) == std::nullopt);
            REQUIRE(g[0]->phase() == Phase(0));
            REQUIRE(g[1]->phase() == Phase(0));
        }
    }
}

TEST_CASE("Adding Edges between Z-X/X-Z connections", "[zx][basic]") {
    ZXGraph g;

    auto [vt1, vt2] = GENERATE(std::make_pair(VertexType::z, VertexType::x),
                               std::make_pair(VertexType::x, VertexType::z));
    g.add_vertex(vt1);
    g.add_vertex(vt2);

    SECTION("Add simple edge") {
        g.add_edge(g[0], g[1], EdgeType::simple);
        REQUIRE(g.get_edge_type(g[0], g[1]) == EdgeType::simple);

        SECTION("Add another simple edge") {
            g.add_edge(g[0], g[1], EdgeType::simple);
            REQUIRE(g.get_edge_type(g[0], g[1]) == std::nullopt);
            REQUIRE(g[0]->phase() == Phase(0));
            REQUIRE(g[1]->phase() == Phase(0));
        }

        SECTION("Add another hadamard edge") {
            g.add_edge(g[0], g[1], EdgeType::hadamard);
            REQUIRE(g.get_edge_type(g[0], g[1]) == EdgeType::hadamard);
            REQUIRE(g[0]->phase() == Phase(1));
            REQUIRE(g[1]->phase() == Phase(0));
        }
    }

    SECTION("Add hadamard edge") {
        g.add_edge(g[0], g[1], EdgeType::hadamard);
        REQUIRE(g.get_edge_type(g[0], g[1]) == EdgeType::hadamard);

        SECTION("Add another simple edge") {
            g.add_edge(g[0], g[1], EdgeType::simple);
            REQUIRE(g.get_edge_type(g[0], g[1]) == EdgeType::hadamard);
            REQUIRE(g[0]->phase() == Phase(1));
            REQUIRE(g[1]->phase() == Phase(0));
        }

        SECTION("Add another hadamard edge") {
            g.add_edge(g[0], g[1], EdgeType::hadamard);
            REQUIRE(g.get_edge_type(g[0], g[1]) == EdgeType::hadamard);
            REQUIRE(g[0]->phase() == Phase(0));
            REQUIRE(g[1]->phase() == Phase(0));
        }
    }
}

TEST_CASE("Adding >1 edge between other vertex types", "[zx][basic]") {
    ZXGraph g;

    auto vt1 = GENERATE(
        VertexType::h_box, VertexType::boundary);
    auto vt2 = GENERATE(
        VertexType::z, VertexType::x);
    auto vt3 = GENERATE(
        VertexType::h_box, VertexType::boundary);

    auto etype12 = GENERATE(EdgeType::simple, EdgeType::hadamard);
    auto etype13 = GENERATE(EdgeType::simple, EdgeType::hadamard);
    auto etype23 = GENERATE(EdgeType::simple, EdgeType::hadamard);

    auto etype_throw = GENERATE(EdgeType::simple, EdgeType::hadamard);

    g.add_vertex(vt1);
    g.add_vertex(vt2);
    g.add_vertex(vt3);
    g.add_edge(g[0], g[1], etype12);
    g.add_edge(g[0], g[2], etype13);
    g.add_edge(g[1], g[2], etype23);

    REQUIRE_THROWS_AS(g.add_edge(g[0], g[1], etype_throw), std::logic_error);
    REQUIRE_THROWS_AS(g.add_edge(g[0], g[2], etype_throw), std::logic_error);
    REQUIRE_THROWS_AS(g.add_edge(g[1], g[2], etype_throw), std::logic_error);
}

TEST_CASE("Adding an edge between a vertex and itself", "[zx][basic]") {
    ZXGraph g;

    auto const vt = GENERATE(VertexType::z, VertexType::x);
    g.add_vertex(vt);

    g.add_edge(size_t{0}, 0, EdgeType::simple);
    REQUIRE(g.num_neighbors(g[0]) == 0);
    REQUIRE(g[0]->phase() == Phase(0));

    g.add_edge(size_t{0}, 0, EdgeType::hadamard);
    REQUIRE(g.num_neighbors(g[0]) == 0);
    REQUIRE(g[0]->phase() == Phase(1));
}

TEST_CASE("Adding an edge between a Z/X vertex and itself", "[zx][basic]") {
    ZXGraph g;

    auto const vt = GENERATE(VertexType::z, VertexType::x);
    g.add_vertex(vt);

    g.add_edge(size_t{0}, 0, EdgeType::simple);
    REQUIRE(g.num_neighbors(g[0]) == 0);
    REQUIRE(g[0]->phase() == Phase(0));

    g.add_edge(size_t{0}, 0, EdgeType::hadamard);
    REQUIRE(g.num_neighbors(g[0]) == 0);
    REQUIRE(g[0]->phase() == Phase(1));
}

TEST_CASE(
    "Adding an edge between a boundary/H-box and itself", "[zx][basic]") {
    ZXGraph g;
    auto const vt = GENERATE(VertexType::h_box, VertexType::boundary);
    g.add_vertex(vt);

    auto const et = GENERATE(EdgeType::simple, EdgeType::hadamard);
    REQUIRE_THROWS_AS(g.add_edge(0ul, 0, et), std::logic_error);
}

// TODO - what is the correct behavior for self loops on H-boxes?

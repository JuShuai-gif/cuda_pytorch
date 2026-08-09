// Correctness checks for eager_vs_lazy: count constructions under each path.

#include <cstdio>
#include <map>
#include <memory>
#include <string>

#include "test_utils.hpp"

namespace {

int g_audio_loads = 0;

struct Audio {
    explicit Audio(std::string path) : name(std::move(path)) {
        ++g_audio_loads;
    }
    std::string name;
};

std::unique_ptr<Audio> load_audio(const std::string& path) {
    return std::make_unique<Audio>(path);
}

class AudioLibrary {
public:
    AudioLibrary() {
        map_.emplace("red_fox", std::make_unique<Audio>("red_fox.wav"));
    }

    auto get_eager(const std::string& id,
                   std::unique_ptr<Audio> otherwise) const {
        auto it = map_.find(id);
        return it != map_.end() ? std::move(it->second) : std::move(otherwise);
    }

    template <typename Fn>
    auto get_lazy(const std::string& id, Fn otherwise) const {
        auto it = map_.find(id);
        return it != map_.end() ? std::move(it->second) : otherwise();
    }

private:
    mutable std::map<std::string, std::unique_ptr<Audio>> map_;
};

}  // namespace

int main() {
    // Hit path: eager constructs the fallback, lazy does not.
    {
        g_audio_loads = 0;
        AudioLibrary library;  // +1 for the map entry
        auto eager_hit = library.get_eager("red_fox", load_audio("d.wav"));
        CHP_CHECK(g_audio_loads == 2);  // map fill + fallback
        CHP_CHECK(eager_hit->name == "red_fox.wav");
    }

    {
        g_audio_loads = 0;
        AudioLibrary library;
        auto lazy_hit = library.get_lazy("red_fox", [] { return load_audio("d.wav"); });
        CHP_CHECK(g_audio_loads == 1);  // map fill only; fallback never built
        CHP_CHECK(lazy_hit->name == "red_fox.wav");
    }

    // Miss path: constructs the fallback.
    {
        g_audio_loads = 0;
        AudioLibrary library;
        auto lazy_miss = library.get_lazy("nope", [] { return load_audio("d.wav"); });
        CHP_CHECK(g_audio_loads == 2);  // map fill + fallback
        CHP_CHECK(lazy_miss->name == "d.wav");
    }

    return chp::test_summary("eager_vs_lazy");
}

// Eager vs lazy evaluation in a library lookup (PDF p.259).
//
// The eager getter constructs the fallback audio file even when the lookup
// succeeds; the lazy getter defers that construction until it is really
// needed by accepting a factory function instead of a value.

#include <cstdio>
#include <map>
#include <memory>
#include <string>

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

    // Eager: the fallback Audio is already constructed when this is called,
    // even if the id is found in the map.
    auto get_eager(const std::string& id,
                   std::unique_ptr<Audio> otherwise) const {
        auto it = map_.find(id);
        return it != map_.end() ? std::move(it->second) : std::move(otherwise);
    }

    // Lazy: the fallback is only constructed if the id is missing.
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
    std::printf("== eager_vs_lazy ==\n");

    {
        AudioLibrary library;
        const auto before = g_audio_loads;
        auto found = library.get_eager("red_fox", load_audio("default_fox.wav"));
        std::printf("get_eager (hit): audio=%s, loads=%d\n",
                    found->name.c_str(), g_audio_loads - before);
    }

    {
        AudioLibrary library;
        const auto before = g_audio_loads;
        auto found = library.get_lazy("red_fox",
                                      [] { return load_audio("default_fox.wav"); });
        std::printf("get_lazy (hit):  audio=%s, loads=%d\n",
                    found->name.c_str(), g_audio_loads - before);
    }

    {
        AudioLibrary library;
        const auto before = g_audio_loads;
        auto missed = library.get_lazy("no_such_id",
                                       [] { return load_audio("default_fox.wav"); });
        std::printf("get_lazy (miss): audio=%s, loads=%d\n",
                    missed->name.c_str(), g_audio_loads - before);
    }

    return 0;
}

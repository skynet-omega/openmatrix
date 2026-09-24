#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace epoch {

struct Port {
  std::string name;
  std::string unit;
};

struct Event {
  std::int64_t at_ns;
  std::uint64_t sequence;
  std::string port;
  double value;
};

class Context {
 public:
  std::int64_t now_ns = 0;
  std::int64_t end_ns = 0;

  void publish(std::int64_t at_ns, const std::string& port, double value);
  double sum_events(const std::string& port, std::int64_t through_ns) const;
  void set_value(const std::string& port, double value);
  double value(const std::string& port) const;
  const std::vector<Event>& events() const { return events_; }

 private:
  friend class Session;
  // Events published during this transaction start here. Prior events stay
  // available for provenance but are not delivered to a later tick again.
  std::size_t event_begin_ = 0;
  std::vector<Event> events_;
  std::map<std::string, double> values_;
  std::uint64_t next_sequence_ = 0;
};

class Owner {
 public:
  virtual ~Owner() = default;
  virtual std::string name() const = 0;
  virtual std::vector<Port> reads() const = 0;
  virtual std::vector<Port> writes() const = 0;
  virtual void snapshot() = 0;
  virtual void advance(Context& context) = 0;
  virtual void commit() = 0;
  virtual void rollback() noexcept = 0;
  virtual void release_snapshot() noexcept = 0;
};

class Session {
 public:
  explicit Session(std::int64_t tick_ns);
  void add(std::unique_ptr<Owner> owner);
  void compile();
  void tick();
  const Context& context() const { return context_; }
  std::int64_t time_ns() const { return context_.now_ns; }

 private:
  std::int64_t tick_ns_;
  bool compiled_ = false;
  Context context_;
  std::vector<std::unique_ptr<Owner>> owners_;
  std::map<std::string, std::string> units_;
};

}  // namespace epoch

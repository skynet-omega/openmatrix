#include "epoch_session.hpp"

#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace epoch {

void Context::publish(std::int64_t at_ns, const std::string& port, double value) {
  if (at_ns < now_ns || at_ns > end_ns || !std::isfinite(value)) {
    throw std::runtime_error("invalid event time or value");
  }
  events_.push_back(Event{at_ns, next_sequence_++, port, value});
}

double Context::sum_events(const std::string& port, std::int64_t through_ns) const {
  if (through_ns < now_ns || through_ns > end_ns) {
    throw std::runtime_error("event query outside current tick");
  }
  double sum = 0.0;
  for (std::size_t i = event_begin_; i < events_.size(); ++i) {
    const auto& event = events_[i];
    if (event.port == port && event.at_ns >= now_ns && event.at_ns <= through_ns) {
      sum += event.value;
    }
  }
  return sum;
}

void Context::set_value(const std::string& port, double value) {
  if (!std::isfinite(value)) throw std::runtime_error("nonfinite channel value");
  values_[port] = value;
}

double Context::value(const std::string& port) const {
  const auto found = values_.find(port);
  if (found == values_.end()) throw std::runtime_error("channel has no current value: " + port);
  return found->second;
}

Session::Session(std::int64_t tick_ns) : tick_ns_(tick_ns) {
  if (tick_ns <= 0) throw std::runtime_error("tick duration must be positive");
}

void Session::add(std::unique_ptr<Owner> owner) {
  if (compiled_) throw std::runtime_error("cannot add owner after compile");
  if (!owner) throw std::runtime_error("null owner");
  owners_.push_back(std::move(owner));
}

void Session::compile() {
  std::set<std::string> owner_names;
  std::map<std::string, std::string> writers;
  for (const auto& owner : owners_) {
    if (owner->name().empty() || !owner_names.insert(owner->name()).second) {
      throw std::runtime_error("empty or duplicate owner name");
    }
    for (const Port& port : owner->writes()) {
      if (port.name.empty() || port.unit.empty()) throw std::runtime_error("incomplete write port");
      if (!writers.emplace(port.name, owner->name()).second) {
        throw std::runtime_error("duplicate port writer: " + port.name);
      }
      units_[port.name] = port.unit;
    }
  }
  for (const auto& owner : owners_) {
    for (const Port& port : owner->reads()) {
      auto found = units_.find(port.name);
      if (found == units_.end() || found->second != port.unit) {
        throw std::runtime_error("missing or unit-incompatible port: " + port.name);
      }
    }
  }
  compiled_ = true;
}

void Session::tick() {
  if (!compiled_) throw std::runtime_error("session not compiled");
  if (context_.now_ns > std::numeric_limits<std::int64_t>::max() - tick_ns_) {
    throw std::runtime_error("time overflow");
  }
  Context before = context_;
  context_.end_ns = context_.now_ns + tick_ns_;
  context_.event_begin_ = context_.events_.size();
  std::size_t snapped = 0;
  try {
    for (auto& owner : owners_) {
      owner->snapshot();
      ++snapped;
    }
    for (auto& owner : owners_) owner->advance(context_);
    for (auto& owner : owners_) owner->commit();
    for (auto& owner : owners_) owner->release_snapshot();
    context_.now_ns = context_.end_ns;
  } catch (...) {
    for (std::size_t i = snapped; i > 0; --i) {
      owners_[i - 1]->rollback();
      owners_[i - 1]->release_snapshot();
    }
    context_ = std::move(before);
    throw;
  }
}

}  // namespace epoch
